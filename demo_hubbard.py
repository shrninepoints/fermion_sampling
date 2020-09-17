import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from State import *
distance = State.distance2D
from copy import deepcopy
import time
from multiprocessing import Pool
import sys
L = 4       # system size (2D, L*L square lattice)
N = 16      # num of electron
U = 0       # take tunneling strength as a unit
disorder = 0
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
    def fs(v,epsilon,M_optim,loop,U,N,L,disorder):
        E = np.zeros(M_optim)
        V = np.zeros(M_optim+1)
        V[0] = v

        #H,impurity = hamiltonian(L)
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]
        N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)

        for ind_optim in range(M_optim):
            np.random.seed()
            '''
            if ind_optim > 4:
                var = np.var(E[ind_optim-4:ind_optim])
                loop = int(1 / var)*10+1000
                loop = np.clip(loop,1000,1000)
                print(loop,end="\r")
            '''
            samples = loop
            energy = d = e_mul_d = normalize = 0
            particle = 0
            #e_list = []
            mat = phi_0 + np.random.rand(sites*2,N) / 10 ** 12     # add small pertubation to avoid singular matrix error
            #print(mat)
            for _ in range(loop):
                sample = sampling(mat,N,2*sites)
                sample.sort()
                state = State(sites,sample[:N-N_down],sample[N-N_down:]-sites)
                doubleNum = state.getDoubleNum()
                e = state.energy(v,U,phi_0) - disorder * (np.dot(state.getState()[0:sites],impurity) + np.dot(state.getState()[sites:],impurity))
                weight = np.exp(-2*v * doubleNum)
                d = d - weight * doubleNum
                energy = energy + weight * e
                normalize = normalize + weight
                #e_list.append(e)
                #weight_list.append(weight)
                if 0 in sample: particle = particle + weight
                if sites in sample: particle = particle + weight
                e_mul_d = e_mul_d - weight * doubleNum * e
            if samples / loop < 0.9: raise ValueError("Too many fails")
            D = d / normalize
            E[ind_optim] = energy / normalize
            #print(np.sum(np.dot(np.array(e_list - E[ind_optim]) ** 2,weight_list) * (normalize / (normalize ** 2 - np.sum(np.array(weight_list)**2)))))
            
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
        #H,impurity = hamiltonian(L)
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]
        N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
        state = State(sites,np.random.choice(sites, np.int(N - N_down), replace=False),np.random.choice(sites, np.int(N_down), replace=False))    # initial state
        # loop for grad descent 
        for ind_optim in range(M_optim):
            '''
            if ind_optim > 4:
                var = np.var(E[ind_optim-4:ind_optim])
                Mtotal = (int(1 / var)+100) * interval + Meq
                Mtotal = np.clip(Mtotal,100,100000)
                print(Mtotal,end="\r")
            '''
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
                    e_list.append(energy)
        
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
            E_mul_D = e_mul_d / mc
            f = 2 * (E[ind_optim]*D - E_mul_D)  # calc gradient
            v = v + epsilon * f # update variational parameter
            v = np.clip(v,-10,10)
            V[ind_optim+1] = v
            
        return tau_list

    @timeit
    def vmc_sr(v,epsilon,M_optim,Meq,Mc,U,N,L):

        interval = sites * 2
        Mtotal = Meq + Mc * interval
        E = np.zeros(M_optim)
        param_num = L if L % 2 == 1 else L+1
        print(param_num)
        V = np.zeros((M_optim+1,param_num))
        V[0] = v

        H = hamiltonian(L,disorder)
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]
        N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
        N_up = N - N_down
        state = State(sites,np.random.choice(sites, np.int(N_up), replace=False),np.random.choice(sites, np.int(N_down), replace=False))    # initial state

        # loop for sr 
        for ind_optim in range(M_optim):

            '''
            if ind_optim > 4:
                var = np.var(E[ind_optim-4:ind_optim])
                Mtotal = (int(1 / var)+100) * interval + Meq
                Mtotal = np.clip(Mtotal,100,100000)
                print(Mtotal,end="\r")
            '''
            W = np.dot(phi_0, np.linalg.inv(phi_0[state.getX()]))
            mc = e = d = e_mul_o = 0
            np.random.seed()
            A = v - v[0]

            site_double = np.intersect1d(state.up, state.down)
            site_occ = np.union1d(state.up, state.down)
            n = np.zeros(sites, dtype=np.int8)
            n[site_occ] = 1
            n[site_double] = 2
            T = np.zeros(sites)
            for i in range(len(site_occ)):
                ind = site_occ[i]
                for j in range(sites):
                    d = distance(L,ind,j)
                    T[ind] = T[ind] + v[d] * n[j]

            o = np.zeros(param_num, dtype=np.int)
            o_mul_o = np.zeros((param_num, param_num), dtype=np.int)

            np.random.seed()
            for _ in range(Mtotal):
            # randomly choose a new configuration
                state_new = deepcopy(state)
                K,l = state_new.randomHopState()
                rl = state.getX()[l]

                J_Ratio = np.exp(T[rl%sites] - T[K%sites] + A[distance(L,rl%sites, K%sites)])
                detRatio = W[K,l]

                # Recept in Markov chain
                r = np.square(np.abs(J_Ratio*detRatio))
                if np.random.rand() < np.min([1, r]):
                    state = state_new
                    # update of Jastrow factor
                    for j in range(sites):
                        d_kj = distance(L, j, K%sites)
                        d_lj = distance(L, j, rl%sites)
                        T[j] = T[j] + v[d_kj] - v[d_lj]
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
                    tmp_o = state.O_k()
                    tmp_e = state.energy_jastrow(v,U,phi_0)
                    o = o - tmp_o
                    e = e + tmp_e
                    e_mul_o = e_mul_o - tmp_o * tmp_e
                    tmp_o_vec = tmp_o.reshape((param_num,1))
                    o_mul_o = o_mul_o + np.dot(tmp_o_vec, tmp_o_vec.T)
                
            O = o / mc
            E[ind_optim] = e / mc
            E_mul_O = e_mul_o / mc
            f = 2 * (E[ind_optim]*O - E_mul_O)
            O_vec = O.reshape((param_num, 1))
            S = o_mul_o / mc - np.dot(O_vec, O_vec.T)
            print(S)
            try:
                dealt_v = np.linalg.solve(S, epsilon * f)
            except:
                S = S + np.random.rand(param_num,param_num) / 10 ** 4
                dealt_v = np.linalg.solve(S, epsilon * f)
            v = v + dealt_v
            v = np.clip(v,-10,10)
            V[ind_optim+1] = v
        return E,V

    @timeit
    def fs_corr(v,loop,U,N,L):
        sites = L**2
        H = hamiltonian(L)
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]

        np.random.seed()
        energy = normalize = 0
        correlation_up = np.zeros((sites, sites))
        correlation_down = np.zeros((sites, sites))
        correlation_pair = np.zeros((sites, sites))
        correlation_n = np.zeros((sites, sites))
        correlation_s = np.zeros((sites, sites))
        mat = phi_0 + np.random.rand(sites*2,N) / 10 ** 12     # add small pertubation to avoid singular matrix error
        for _ in range(loop):
            sample = sampling(mat,N,2*sites)
            sample.sort()
            N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
            state = State(sites,sample[:N-N_down],sample[N-N_down:]-sites)
            weight = np.exp(-2*v * state.getDoubleNum())
            energy = energy + weight * state.energy(v,U,phi_0)
            normalize = normalize + weight
            a,b = state.correlation_c(v,phi_0)
            correlation_up = correlation_up + a * weight
            correlation_down = correlation_down + b * weight
            correlation_pair = correlation_pair + state.correlation_pair(v,phi_0) * weight
            a,b = state.correlation_ns(v,phi_0)
            correlation_n = correlation_n + a * weight
            correlation_s = correlation_s + b * weight
        print("normal = ",normalize)
        Correlation_up = correlation_up / normalize
        Correlation_down = correlation_down / normalize
        Correlation_pair = correlation_pair / normalize
        Correlation_n = correlation_n / normalize
        Correlation_s = correlation_s / normalize
        return Correlation_up, Correlation_down, Correlation_pair, Correlation_n, Correlation_s

    @timeit
    def vmc_corr(v,Meq,Mc,U,N,L):
        sites = L**2
        interval = sites * 2
        Mtotal = Meq + Mc * interval

        H = hamiltonian(L)
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]
        N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
        N_up = N - N_down
        state = State(sites,np.random.choice(sites, np.int(N_up), replace=False),np.random.choice(sites, np.int(N_down), replace=False))    # initial state
        
        W = np.dot(phi_0, np.linalg.inv(phi_0[state.getX()]))
        mc = e = 0
        correlation_up = np.zeros((sites, sites))
        correlation_down = np.zeros((sites, sites))
        correlation_pair = np.zeros((sites, sites))
        correlation_n = np.zeros((sites, sites))
        correlation_s = np.zeros((sites, sites))
        np.random.seed()
        # loop for VMC    
        for _ in range(Mtotal):
            # randomly choose a new configuration
            state_new = deepcopy(state)
            K,l = state_new.randomHopState()
            detRatio = W[K,l] # Recept in Markov chain
            J_Ratio = np.exp(-v * (state_new.getDoubleNum() - state.getDoubleNum())) # update of Jastrow factor
            r = np.square(np.abs(J_Ratio*detRatio))
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
                a,b = state.correlation_c(v,phi_0)
                correlation_up = correlation_up + a
                correlation_down = correlation_down + b
                correlation_pair = correlation_pair + state.correlation_pair(v,phi_0)
                a,b = state.correlation_ns(v,phi_0)
                correlation_n = correlation_n + a
                correlation_s = correlation_s + b

        Correlation_up = correlation_up / mc
        Correlation_down = correlation_down / mc
        Correlation_pair = correlation_pair / mc
        Correlation_n = correlation_n / mc
        Correlation_s = correlation_s / mc       

        return Correlation_up, Correlation_down, Correlation_pair, Correlation_n, Correlation_s

    def corr():
        method = ["fs"]
        if "fs" in method: 
            v = 1      
            loop = 1000
            Correlation_up, Correlation_down, Correlation_pair, Correlation_n, Correlation_s = fs_corr(v,loop,U,N,L)

        if "vmc" in method: 
            v = 1      
            Meq = 1000
            Mc = 100
            Correlation_up, Correlation_down, Correlation_pair, Correlation_n, Correlation_s = vmc_corr(v,Meq,Mc,U,N,L)

        print(np.sum(np.diag(Correlation_up)))
        print(np.sum(np.diag(Correlation_down)))
        print(np.sum(np.diag(Correlation_pair)))
        print(np.diag(Correlation_n))    
        plt.figure(); plt.imshow(Correlation_pair); plt.colorbar()
        plt.figure(); plt.imshow(Correlation_down); plt.colorbar()
        plt.show()

    def optimize():
        def analysis_data(E,theo_mean):
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
            v = 0      # initial variational parameter
            epsilon = 0   # variational step length
            M_optim = 16    # num of variational steps
            loop = 100
            E,V = fs(v,epsilon,M_optim,loop,U,N,L,disorder)
            print(E)
            print(V)
            fsdata = analysis_data(E,theo_occupation)
            print(fsdata)
        if "vmc" in method: 
            print("VMC start")
            v = 0  # initial variational parameter
            epsilon = 0  # variational step length
            M_optim = 16    # num of variational steps
            Meq = 1000      # vmc step to reach near ground state
            Mc = 100        # vmc step to accumulate observable
            E,V = vmc(v,epsilon,M_optim,Meq,Mc,U,N,L,disorder)
            print(E)
            print(V)
            vmcdata = analysis_data(E,theo_occupation)
            print(vmcdata)
        if "tau" in method:
            print("tau start")
            v = 0  # initial variational parameter
            epsilon = 0  # variational step length
            M_optim = 8    # num of variational steps
            Meq = 10000      # vmc step to reach near ground state
            Mc = 10000        # vmc step to accumulate observable
            tau_list = vmc(v,epsilon,M_optim,Meq,Mc,U,N,L,disorder)
            tau = np.sum(tau_list) / len(tau_list)
            tau_error = np.sqrt(np.sum(np.array(tau_list - tau)**2) / len(tau_list))

        #return [tau,tau_error]
        return list(vmcdata) + list(fsdata)

    H = hamiltonian(L,disorder,impurity=[-0.82125305923681946, -0.6877573621434765, 0.9593998884827919, 0.26680609116515996, 0.6871452038204888, -0.11202607110398066, 0.48901813063507005, -0.8187008605448085, -0.2820690654939455, -0.7967271456947933, -0.8683407160247776, -0.4327277119621351, -0.037541526913182244, 0.2986313575050994, 0.24362386763614174, -0.854617852342691])
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
    #file = open("data/corr.txt", 'w'); sys.stdout = file
    with Pool(10) as p:
        tau_list = p.map(main, [0.1*i for i in range(1)])
    for tau in tau_list:
        print(str(tau)[1:-1])