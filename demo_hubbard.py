import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from State import *
import slater_reconstruction_spin
distance = State.distance2D
from copy import deepcopy
import time
from multiprocessing import Pool
import sys

L = 3       # system size (2D, L*L square lattice)
N = 8      # num of electron
U = 5       # take tunneling strength as a unit
disorder = 0.0001
sites = L**2

def main(disorder):
    def timeit(f):
        def timed(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            print('func:%r args:[%r, %r] took: %2.4f sec' % \
            (f.__name__, args[0:2], kw, te-ts))
            return result
        return timed

    def hamiltonian(L,disorder,U = 0,impurity = None):
        sites = L ** 2
        H1 = np.zeros((sites,sites))
        for i in range(sites-1):
            H1[i,i+1] = -1
            H1[i,(i+L)%sites] = -1
        for j in range(1,L):
            H1[j*L-1, j*L] = 0
            H1[j*L-1, (j-1)*L] = -1
        H1[sites-1, L-1] = -1
        H1[sites-1, sites-L] = -1
        H1 = H1 + H1.T
        np.random.seed()
        if impurity == None:
            impurity = (np.random.rand(sites) - 0.5) * 2
        H = np.kron(np.identity(2), H1) + disorder * np.diag(np.append(impurity,impurity))
        return H

    def sampling(U,N,L):
        # dealing with matrix that changed a row / collum, return new det with O(n)
        def detUpdate(delta_v,delta_index,U_inv,U_det,axis):  # detUpdate result could be minus det, but we don't care here                                   
            if axis == 0:
                detRatio = 1 + np.dot(U_inv[:,delta_index],delta_v)
            elif axis == 1:
                detRatio = 1 + np.dot(U_inv[delta_index,:],delta_v) # delta_v input can be either row or column vector
            return detRatio * U_det     # return value is a complex number
        
        # following update methods dealing with enlarged matrix with row and collum added at the end
        # update the inv and det of enlarged matrix with O(n^2)
        def matrixDetUpdate(m_new,m_det,m_inv,n): # m_new is n*n, m is (n-1)*(n-1)
            v_new = m_new[0:-1,-1]
            delta_v_mat = np.array([v_new,]*(n-1)).T - m_new[0:-1,0:-1]
            m_det_list = np.array([detUpdate(delta_v_mat[:,i],i,m_inv,m_det,1) for i in range(n-1)] + [(-1 ** (n-1))*m_det])     # only care about relative +-
            m_new_det = np.dot(m_new[-1],m_det_list)
            return m_new_det
        
        def matrixInvUpdate(m_new,m_inv,n): # m_new is n*n, m is (n-1)*(n-1)
            u = np.expand_dims(m_new[0:-1,-1],axis=1)
            v = np.expand_dims(m_new[-1,0:-1],axis=0)
            c = m_new[-1,-1]
            k_inv = 1 / (c - v.dot(m_inv).dot(u))
            m_new_inv = m_inv + k_inv * np.dot(m_inv.dot(u),v.dot(m_inv))
            m_new_inv = np.c_[m_new_inv,-k_inv * m_inv.dot(u)]
            m_new_inv = np.r_[m_new_inv,np.expand_dims(np.append(-k_inv * v.dot(m_inv), np.array([k_inv])),axis=0)]
            return m_new_inv

        v = np.random.permutation(N)
        x = []              # x = [x1,...xk], xi = 0...L-1
        for i in range(N):  # k = i + 1
            p = np.zeros(L,dtype='float128')
            first_matrix = None
            for j in range(L):            
                if j not in x: 
                    if first_matrix is None:
                        j0 = j
                        first_matrix = U[np.ix_(x+[j],v[0:i+1])] # TODO: use "add row / column to replace np.ix_"
                        if i == 0: # the very first matrix, O(1)
                            first_matrix_inv = np.linalg.inv(first_matrix)
                            first_matrix_det = np.linalg.det(first_matrix)
                        else:   # the first matrix of new rank, O(n^2)
                            first_matrix_inv = matrixInvUpdate(first_matrix,prev_matrix_inv,i+1)
                            first_matrix_det = matrixDetUpdate(first_matrix,prev_matrix_det,prev_matrix_inv,i+1)
                        p[j] = np.square(np.abs(first_matrix_det))
                    else: # update det and use it, O(n)
                        delta_v = U[j,v[0:i+1]] - U[j0,v[0:i+1]]
                        p[j] = np.square(np.float128(np.abs(detUpdate(delta_v,-1,first_matrix_inv,first_matrix_det,0))))
                        # use of higher accuracy is necessary, otherwise introduce divide by 0 for small first_matrix_det
            p = np.float64(p / sum(p))
            x.append(np.random.choice(L,p = p))    # choose position wrt p 
            if i == 0:  # the choosen matrix, either the very first one or of new ranks, O(n^2)
                choosen_matrix = np.array([[U[x[0],v[0]]]])
                prev_matrix_inv = np.array([[1/choosen_matrix[0,0]]])
                prev_matrix_det =  choosen_matrix[0,0]
            else:
                choosen_matrix = np.r_[first_matrix[0:-1], np.expand_dims(U[x[-1],v[0:i+1]],axis=0)]
                prev_matrix_det = matrixDetUpdate(choosen_matrix,prev_matrix_det,prev_matrix_inv,i+1)
                prev_matrix_inv = matrixInvUpdate(choosen_matrix,prev_matrix_inv,i+1) # TODO: this can be optimized using detUpdate
        return np.array(x)

    @timeit
    def fs(v,epsilon,M_optim,loop,U,N,L,disorder,phi_1):
        E = np.zeros(M_optim)
        V = np.zeros(M_optim+1)
        V[0] = v

        #H,impurity = hamiltonian(L)
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]
        N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)

        for ind_optim in range(M_optim):
            np.random.seed()
            energy = d = e_mul_d = normalize = 0
            particle = 0
            e_list = []
            mat = phi_1 + np.random.rand(sites*2,N) / 10 ** 12     # add small pertubation to avoid singular matrix error
            for _ in range(loop):
                sample = sampling(mat,N,2*sites)
                sample.sort()
                state = State(sites,sample[:N-N_down],sample[N-N_down:]-sites)
                doubleNum = state.getDoubleNum()
                e = state.energy(v,U,phi_1) - disorder * (np.dot(state.getState()[0:sites],impurity) + np.dot(state.getState()[sites:],impurity))
                weight = np.exp(-2*v * doubleNum)
                d = d - weight * doubleNum
                energy = energy + weight * e
                normalize = normalize + weight
                e_list.append(e)
                #weight_list.append(weight)
                if 0 in sample: particle = particle + weight
                if sites in sample: particle = particle + weight
                e_mul_d = e_mul_d - weight * doubleNum * e

            D = d / normalize
            E[ind_optim] = energy / normalize      
            print("fs e = ",energy)      
            E_mul_D = e_mul_d / normalize
            f = 2 * (E[ind_optim]*D - E_mul_D)
            v = v + epsilon * f
            v = np.clip(v,-10,10)
            V[ind_optim+1] = v
        return E,V

    @timeit
    def vmc(v,epsilon,M_optim,Meq,Mc,U,N,L,disorder):
        interval = N
        Mtotal = Meq + Mc * interval
        E = np.zeros(M_optim)
        V = np.zeros(M_optim+1)
        V[0] = v
        tau_list = []
        
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]
        N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
        state = State(sites,np.random.choice(sites, np.int(N - N_down), replace=False),np.random.choice(sites, np.int(N_down), replace=False))    # initial state
        # loop for grad descent 
        for ind_optim in range(M_optim):
            W = np.dot(phi_0, np.linalg.inv(phi_0[state.getX()]))
            mc = energy = d = e_mul_d = 0
            particle = 0
            np.random.seed()
            # loop for VMC    
            e_list = []
            for _ in range(Mtotal):
                # randomly choose a new configuration
                state_new = deepcopy(state)
                K,l = state_new.randomHopState()
                detRatio = W[K,l] # Recept in Markov chain
                J_Ratio = np.exp(-v * (state_new.getDoubleNum() - state.getDoubleNum())) # update of Jastrow factor
                r = np.square(np.abs(detRatio * J_Ratio))
                if np.random.rand() < np.min([1, r]):
                    state = state_new
                    # update of determinant
                    W_Il = W[:, l].reshape(2*sites,1) * np.ones_like(W)
                    W_Kj = W[K] * np.ones_like(W)
                    W_Il_delta_lj = np.zeros_like(W)
                    W_Il_delta_lj[:, l] = W[:, l]
                    W_new = W - W_Il * W_Kj / W[K,l] + W_Il_delta_lj / W[K,l]
                    W = W_new

                # extract the results
                if _ >= Meq and (_ - Meq) % interval == 0:
                    mc += 1
                    d = d - state.getDoubleNum()
                    energy = energy + state.energy(v,U,phi_0) - disorder * (np.dot(state.getState()[0:sites],impurity) + np.dot(state.getState()[sites:],impurity))
                    e_mul_d = e_mul_d - state.getDoubleNum() * state.energy(v,U,phi_0)
                    p = 0
                    if 0 in state.getX(): p = p + 1
                    if sites in state.getX(): p = p + 1
                    particle = particle + p
                    e = state.energy(v,U,phi_0)
                    if e > 100: print(state.getX())
                    e_list.append(e)
        
            def correlation_length(e_list):
                def autocorrelation(step,e_list):
                    assert len(e_list[step:]) == len(e_list[0:len(e_list)-step])
                    cov = np.cov(e_list[step:],e_list[0:len(e_list)-step])[0][1]
                    return cov / np.var(e_list)
                ac_list = []
                for step in range(900):
                    ac_list.append(autocorrelation(step,e_list))
                #print("tau = ",abs(np.polyfit(np.log(np.abs(ac_list[0:20])), np.arange(20), 1)[0])) # corr length / (sites*2)
                tau = np.sum(ac_list)
                #print(disorder, "integrated tau = ", tau)     
                #plt.figure(); plt.plot(ac_list); plt.show() 
                return tau
            #tau_list.append(correlation_length(e_list))
            
            D = d / mc
            E[ind_optim] = energy / mc
            print("vmc e = ", energy)
            E_mul_D = e_mul_d / mc
            f = 2 * (E[ind_optim]*D - E_mul_D)  # calc gradient
            v = v + epsilon * f # update variational parameter
            v = np.clip(v,-10,10)
            V[ind_optim+1] = v    
        return E,V

    def optimize():
        def analysis_data(E,theo_mean = None):
            if theo_mean is None:
                mean = np.sum(E) / len(E)
                error = np.sqrt(np.sum(np.array(E - mean)**2) / len(E))
                return mean,error
            else:
                error = np.sqrt(np.sum(np.array(E - theo_mean)**2) / len(E))
                assert np.sqrt(len(E)) % 1 < 0.01
                trials = len(E)
                piles = int(np.sqrt(trials))
                error_list = []
                for i in range(piles):
                    tmp = E[i*piles:(i*piles + piles)]
                    error_list.append(np.sqrt(np.sum(np.array(tmp - theo_mean)**2) / len(tmp)))
                error_on_error = np.sqrt(np.sum(np.array(error_list - error)**2) / len(error_list))
                return (error,error_on_error)

        method = ["fs","vmc"]
        if "fs" in method: 
            print("FS start")
            v = 0.2      # initial variational parameter
            epsilon = 0.03   # variational step length
            M_optim = 32    # num of variational steps
            loop = 100
            phi_1 = slater_reconstruction_spin.preprocess(phi_0.T,v,N,L**2).T 
            #phi_1 = phi_0
            E,V = fs(v,epsilon,M_optim,loop,U,N,L,disorder,phi_1)
            print(E)
            print(V)
            fsdata = analysis_data(E)
            print(fsdata)
        if "vmc" in method: 
            print("VMC start")
            v = 0.8  # initial variational parameter
            epsilon = 0.03  # variational step length
            M_optim = 16    # num of variational steps
            Meq = 1000      # vmc step to reach near ground state
            Mc = 100        # vmc step to accumulate observable
            E,V = vmc(v,epsilon,M_optim,Meq,Mc,U,N,L,disorder)
            print(E)
            print(V)
            vmcdata = analysis_data(E)
            print(vmcdata)
        if "tau" in method:
            print("tau start")
            v = 0  # initial variational parameter
            epsilon = 0  # variational step length
            M_optim = 4    # num of variational steps
            Meq = 10000      # vmc step to reach near ground state
            Mc = 10000        # vmc step to accumulate observable
            tau_list = vmc(v,epsilon,M_optim,Meq,Mc,U,N,L,disorder)
            tau = np.sum(tau_list) / len(tau_list)
            tau_error = np.sqrt(np.sum(np.array(tau_list - tau)**2) / len(tau_list))

        #return [tau,tau_error]
        return list(vmcdata) + list(fsdata)

    H = hamiltonian(L,disorder)#,impurity=[-0.82125305923681946, -0.6877573621434765, 0.9593998884827919, 0.26680609116515996, 0.6871452038204888, -0.11202607110398066, 0.48901813063507005, -0.8187008605448085, -0.2820690654939455, -0.7967271456947933, -0.8683407160247776, -0.4327277119621351, -0.037541526913182244, 0.2986313575050994, 0.24362386763614174, -0.854617852342691])
    np.savetxt('data/hamiltonian.txt', H, fmt='%.18e')
    #H = np.loadtxt('data/hamiltonian.txt',dtype=float)
    impurity = np.diag(H-hamiltonian(L,0))[:sites]/disorder
    if disorder == 0: impurity = np.zeros(sites)
    print("impurity = ", list(impurity))
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    theo_occupation = np.sum(phi_0[0] ** 2) + np.sum(phi_0[sites] ** 2)
    print("theo_occupation = ", theo_occupation)

    a = optimize()
    return [disorder] + a + [theo_occupation]

if __name__ == "__main__":
    parallel = False
    if parallel:
        #file = open("data/corr.txt", 'w'); sys.stdout = file
        with Pool(10) as p:
            tau_list = p.map(main, [0.1*i for i in range(1)])
        for tau in tau_list:
            print(str(tau)[1:-1])
    else:
        main(disorder)