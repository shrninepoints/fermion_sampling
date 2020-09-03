import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from State import *
from copy import deepcopy
import time

L = 5       # system size (2D, L*L square lattice)
N = 4      # num of electron
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
        s = np.zeros(sites * 2)
        e_min = 0
        mat = phi_0 + np.random.rand(sites*2,N) / 10 ** 8     # add small pertubation to avoid singular matrix error
        for _ in range(loop):
            try:
                sample = sampling(mat,N,2*sites)
                sample.sort()
                state = State(sites,sample)
                d = d - state.getDoubleNum()
                energy = energy + np.exp(-2*v * state.getDoubleNum()) * state.energy(v,U,phi_0)
                normalize = normalize + np.exp(-2*v * state.getDoubleNum())
                #print(state.energy(v,U,phi_0))
                s = s + np.dot(np.exp(-2*v * state.getDoubleNum()), state.getState())
                e_min = np.min([e_min,state.energy(v,U,phi_0)])
                e_mul_d = e_mul_d - state.getDoubleNum() * state.energy(v,U,phi_0)
            except ValueError:
                samples = samples - 1
        if samples / loop < 0.9: raise ValueError("Too many fails")
        print("e_min = ", e_min)
        D = d / samples
        E[ind_optim] = energy/normalize
        print(s / normalize)
        E_mul_D = e_mul_d / samples
        f = 2 * (E[ind_optim]*D - E_mul_D)
        v = v + epsilon * f
        v = np.clip(v,-10,10)
        V[ind_optim+1] = v
    return E,V, s/normalize

@timeit
def vmc(v,epsilon,M_optim,Meq,Mc,U,N,L):
    interval = sites*2
    Mtotal = Meq + Mc * interval
    E = np.zeros(M_optim)
    V = np.zeros(M_optim+1)
    V[0] = v

    H = hamiltonian(L)
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
    N_up = N - N_down
    state = State(sites,np.random.choice(sites, np.int(N_up), replace=False),np.random.choice(sites, np.int(N_down), replace=False))    # initial state
    
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
        s = np.zeros(sites * 2)
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
                d = d - state.getDoubleNum()
                e = e + state.energy(v,U,phi_0)
                #print(state.energy(v,U,phi_0))
                s = s + state.getState()
                #print(state.getState())
                e_mul_d = e_mul_d - state.getDoubleNum() * state.energy(v,U,phi_0)
        D = d / mc
        E[ind_optim] = e / mc
        E_mul_D = e_mul_d / mc
        print(s / mc)
        f = 2 * (E[ind_optim]*D - E_mul_D)  # calc gradient
        v = v + epsilon * f # update variational parameter
        v = np.clip(v,-10,10)
        V[ind_optim+1] = v
    return E,V, s/mc
def cosine(a,b): # cosine similarity between two vectors
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

if __name__ == "__main__":
    method = ["vmc"]
    if "fs" in method: 
        v = 0.448      # initial variational parameter
        epsilon = 0.1   # variational step length
        M_optim = 1    # num of variational steps
        loop = 10000
        E,V,state1 = fs(v,epsilon,M_optim,loop,U,N,L)
    if "vmc" in method: 
        v = 0.448      # initial variational parameter
        epsilon = 0.1   # variational step length
        M_optim = 1    # num of variational steps
        Meq = 1000      # vmc step to reach near ground state
        Mc = 10000        # vmc step to accumulate observable
        E,V,state2 = vmc(v,epsilon,M_optim,Meq,Mc,U,N,L)
    #print(cosine(state1,state2))
    print(E)
    print("Energy error = ", np.sqrt(np.var(E[-5:])))
    print(V)
    plt.figure()
    plt.plot(E, '+')
    plt.figure()
    plt.plot(V, '+')
    #plt.show()