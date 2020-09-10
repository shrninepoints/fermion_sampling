import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from State import *
distance = State.distance2D
from copy import deepcopy
import time

L = 6       # system size (2D, L*L square lattice)
N = 36      # num of electron
U = 4       # take tunneling strength as a unit
sites = L**2

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return timed

def hamiltonian(L,U = 0):
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
    H = np.kron(np.identity(2), H1)
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
        if len(x) > 400:
            print(x)
            plt.figure(); plt.plot(p); plt.show()
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
def fs(v,epsilon,M_optim,loop,U,N,L):
    E = np.zeros(M_optim)
    V = np.zeros(M_optim+1)
    V[0] = v

    H = hamiltonian(L)
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
        e_list = []
        weight_list = []
        mat = phi_0 + np.random.rand(sites*2,N) / 10 ** 12     # add small pertubation to avoid singular matrix error
        #print(mat)
        for _ in range(loop):
            try:
                sample = sampling(mat,N,2*sites)
                sample.sort()
                state = State(sites,sample[:N-N_down],sample[N-N_down:]-sites)
                doubleNum = state.getDoubleNum()
                e = state.energy(v,U,phi_0)
                weight = np.exp(-2*v * doubleNum)
                d = d - weight * doubleNum
                energy = energy + weight * e
                normalize = normalize + weight
                e_list.append(e)
                weight_list.append(weight)
                e_mul_d = e_mul_d - weight * doubleNum * e
            except ValueError:
                samples = samples - 1
        if samples / loop < 0.9: raise ValueError("Too many fails")
        D = d / normalize
        E[ind_optim] = energy/normalize
        #print(normalize)
        print(np.sum(np.dot(np.array(e_list - E[ind_optim]) ** 2,weight_list) * (normalize / (normalize ** 2 - np.sum(np.array(weight_list)**2)))))
        E_mul_D = e_mul_d / normalize
        f = 2 * (E[ind_optim]*D - E_mul_D)
        v = v + epsilon * f
        v = np.clip(v,-10,10)
        V[ind_optim+1] = v
    return E,V

@timeit
def vmc(v,epsilon,M_optim,Meq,Mc,U,N,L):
    interval = sites * 2
    Mtotal = Meq + Mc * interval
    E = np.zeros(M_optim)
    V = np.zeros(M_optim+1)
    V[0] = v

    H = hamiltonian(L)
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
        mc = e = d = e_mul_d = 0
        np.random.seed()
        # loop for VMC    
        e_list = []
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
                d = d - state.getDoubleNum()
                e = e + state.energy(v,U,phi_0)
                e_list.append(state.energy(v,U,phi_0))
                e_mul_d = e_mul_d - state.getDoubleNum() * state.energy(v,U,phi_0)
        #plt.figure(); plt.hist(e_list,bins=20); plt.show()
        D = d / mc
        print(np.var(e_list))
        E[ind_optim] = e / mc
        E_mul_D = e_mul_d / mc
        f = 2 * (E[ind_optim]*D - E_mul_D)  # calc gradient
        v = v + epsilon * f # update variational parameter
        v = np.clip(v,-10,10)
        V[ind_optim+1] = v
    return E,V

@timeit
def vmc_sr(v,epsilon,M_optim,Meq,Mc,U,N,L):

    interval = sites * 2
    Mtotal = Meq + Mc * interval
    E = np.zeros(M_optim)
    param_num = L if L % 2 == 1 else L+1
    print(param_num)
    V = np.zeros((M_optim+1,param_num))
    V[0] = v

    H = hamiltonian(L)
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
    samples = loop
    energy = d = normalize = 0
    e_list = []
    weight_list = []
    correlation_up = np.zeros((sites, sites))
    correlation_down = np.zeros((sites, sites))
    correlation_pair = np.zeros((sites, sites))
    correlation_n = np.zeros((sites, sites))
    correlation_s = np.zeros((sites, sites))
    mat = phi_0 + np.random.rand(sites*2,N) / 10 ** 12     # add small pertubation to avoid singular matrix error
    #print(mat)
    for _ in range(loop):
        try:
            sample = sampling(mat,N,2*sites)
            sample.sort()
            N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
            state = State(sites,sample[:N-N_down],sample[N-N_down:]-sites)
            weight = np.exp(-2*v * state.getDoubleNum())
            d = d - weight * state.getDoubleNum()
            energy = energy + weight * state.energy(v,U,phi_0)
            normalize = normalize + weight
            e_list.append(state.energy(v,U,phi_0))
            weight_list.append(weight)
            a,b = state.correlation_c(v,phi_0)
            correlation_up = correlation_up + a * weight
            correlation_down = correlation_down + b * weight
            correlation_pair = correlation_pair + state.correlation_pair(v,phi_0) * weight
            a,b = state.correlation_ns(v,phi_0)
            correlation_n = correlation_n + a * weight
            correlation_s = correlation_s + b * weight
        except ValueError:
            samples = samples - 1
    if samples / loop < 0.9: raise ValueError("Too many fails")
    Correlation_up = correlation_up / normalize
    Correlation_down = correlation_down / normalize
    Correlation_pair = correlation_pair / normalize
    Correlation_n = correlation_n / normalize
    Correlation_s = correlation_s / normalize
    # D = d / normalize
    # E[ind_optim] = energy/normalize
    # print(normalize)
    # print(np.sum(np.dot(np.array(e_list - E[ind_optim]) ** 2,weight_list) * (normalize / (normalize ** 2 - np.sum(np.array(weight_list)**2))))) 
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
    method = ["fs"]
    if "fs" in method: 
        v = -1      # initial variational parameter
        epsilon = 0.12   # variational step length
        M_optim = 16    # num of variational steps
        loop = 100
        E,V = fs(v,epsilon,M_optim,loop,U,N,L)
    if "vmc" in method: 
        v = 1     # initial variational parameter
        epsilon = 0.12   # variational step length
        M_optim = 16    # num of variational steps
        Meq = 1000      # vmc step to reach near ground state
        Mc = 100        # vmc step to accumulate observable
        E,V = vmc(v,epsilon,M_optim,Meq,Mc,U,N,L)
    if "sr" in method:
        v = np.array([4.665482 ,1.04390547 ,1.08708198])
        epsilon = 0.001   # variational step length
        M_optim = 16    # num of variational steps
        Meq = 1000      # vmc step to reach near ground state
        Mc = 1000        # vmc step to accumulate observable
        E,V = vmc_sr(v,epsilon,M_optim,Meq,Mc,U,N,L)

    #print(cosine(state1,state2))
    print(E)
    print(V)
    print("E error = ", np.var(E))
    print("E mean = ",np.mean(E))
    plt.figure()
    plt.plot(E, '+')
    plt.figure()
    plt.plot(V, '+')
    plt.show()

if __name__ == "__main__":
    corr()