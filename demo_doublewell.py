'''
1d double well potential
'''

import numpy as np
import time
import matplotlib.pyplot as plt

a = 2 ** 6
b = 2 ** 8
c = 11.3146340707317 # V = ax^4 - bx^2 - cx  #12,48 -> 5.093, 24,96 -> 6.978545, 96,384 -> 13.856471078 2**18,2**16->738.8916015625
xrange = 2 # position takes value from [-xrange,xrange]
L = 64
N = 4
interval = xrange / (L-1) * 2

def hamiltonian(a,b,c,plot=False):
    T = np.zeros((L,L))
    V = np.diag(np.array([a*(-xrange+i*interval)**4-b*(-xrange+i*interval)**2-c*(-xrange+i*interval) for i in range(L)]))
    if plot: plt.figure(); plt.plot(np.diag(V)); plt.show()
    for i in range(L):
        for j in range(L):
            if i==j: T[i,j] = (np.pi/interval)**2 / 6
            else: T[i,j] = (-1) ** (i-j) / ((i-j)**2 * interval ** 2)
    return T,V

def energy(x,phi_0,T,V):
    def onehop(x_in):
        result_hop = []
        # left hop
        for i in range(len(x_in)):
            if x_in[i] == 0:
                if L-1 in x_in:
                    continue
                else:
                    left = x_in.copy()
                    left[i] = L-1
                    result_hop.append(left)    
            elif x_in[i] - 1 in x_in:
                continue
            else:
                left = x_in.copy()
                left[i] = left[i] - 1
                result_hop.append(left)
        # right hop
        for i in range(len(x_in)):
            if x_in[i] == L-1:
                if 0 in x_in:
                    continue
                else:
                    right = x_in.copy()
                    right[i] = 0
                    result_hop.append(right)
            elif x_in[i] + 1 in x_in:
                continue
            else:
                right = x_in.copy()
                right[i] = right[i] + 1
                result_hop.append(right)
        return result_hop
    def p2s(position_list): # position index to state
        return np.array([1 if i in position_list else 0 for i in range(L)])
    #print(x)
    psi_x = np.linalg.det(phi_0[x])
    vx = np.sum(np.dot(np.diag(V),p2s(x)))
    x_prime_list = onehop(x)
    psi_x_prime = np.array([np.linalg.det(phi_0[x_prime]) for x_prime in x_prime_list])
    tx = np.array([p2s(x).dot(np.dot(T,p2s(x_prime))) for x_prime in x_prime_list])
    E0 = (np.sum(tx * psi_x_prime) + vx*psi_x) / psi_x
    #print(E0)
    return E0

def fs(loop):
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
    diff = 0
    np.random.seed()
    for _ in range(loop):
        state = sampling(phi_0,N,L)
        diff += N - 2*sum(x < L/2 for x in state)
    return diff/loop


def mc(Meq,Mc):
    def randomHopState(x):
        index_change = np.random.randint(N)
        index_new = np.random.randint(L-N)
        x[index_change] = np.delete(np.arange(L), x)[index_new]
        K = x[index_change]
        return (K,index_change)

    state = np.random.choice(L, N, replace=False)
    W = np.dot(phi_0, np.linalg.inv(phi_0[state]))
    mc = e = diff = 0
    position = np.zeros(L)
    np.random.seed()
    Mtotal = Meq + Mc * N
    diff_list = []
    for _ in range(Mtotal):
        # randomly choose a new configuration
        state_new = state.copy()
        K,l = randomHopState(state_new)
        detRatio = W[K,l] # Recept in Markov chain
        r = np.square(np.abs(detRatio))
        if np.random.rand() < np.min([1, r]):
            state = state_new
            # update of determinant
            W_Il = W[:, l].reshape(L,1) * np.ones_like(W)
            W_Kj = W[K] * np.ones_like(W)
            W_Il_delta_lj = np.zeros_like(W)
            W_Il_delta_lj[:, l] = W[:, l]
            W_new = W - W_Il * W_Kj / W[K,l] + W_Il_delta_lj / W[K,l]
            W = W_new
        # extract the results
        if _ >= Meq and (_ - Meq) % N == 0:
            mc += 1
            position += np.array([1 if i in state else 0 for i in range(L)])
            diff += N - 2*sum(x < L/2 for x in state)
            diff_list.append(N - 2*sum(x < L/2 for x in state))
    diff = diff / mc
    position = position/mc

    def correlation_length():
        def autocorrelation(step,e_list):
            assert len(e_list[step:]) == len(e_list[0:len(e_list)-step])
            cov = np.cov(e_list[step:],e_list[0:len(e_list)-step])[0][1]
            return cov / np.var(e_list)
        ac_list = []
        for step in range(90):
            ac_list.append(autocorrelation(step,diff_list))
        #print(ac_list)
        print("tau = ",abs(np.polyfit(np.log(np.abs(ac_list[0:20])), np.arange(20), 1)[0])) # corr length / (sites*2)
        print("integrated tau = ", np.sum(ac_list))        
        plt.figure(); plt.plot(ac_list); plt.show()   
    correlation_length()
    #plt.figure(); plt.plot(position); plt.show()
    return diff

def search():
    m = 100
    n = 0
    recursion = 0
    while True:
        recursion += 1
        T,V = hamiltonian(a,b,(m+n)/2,plot=False)
        H = T + V
        e, phi = np.linalg.eigh(H)
        phi_0 = phi[: , np.argsort(e)[:N]]
        position = [np.sum(phi_0[i] ** 2) for i in range(L)]
        diff = N - 2*np.sum(position[:L//2])
        if diff < 0.5: n = (m+n)/2
        elif diff > 1.5: m = (m+n)/2
        else: break
        if recursion > 1024: break
    return (m+n)/2


if __name__ == "__main__":
    T,V = hamiltonian(a,b,c,plot=False)
    H = T + V
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    position = [np.sum(phi_0[i] ** 2) for i in range(L)]
    #plt.figure(); plt.plot(position); plt.show()
    diff = N - 2*np.sum(position[:L//2])
    print("theo diff = ", diff)
    s = 0
    if s == 1: 
        print(search())
    else:
        mc_list = []
        fs_list = []
        for i in range(8):
            mc_list.append(mc(1000,1000))
            fs_list.append(fs(1000))
        mc_error = np.sqrt(np.sum(np.array(mc_list - diff)**2) / len(mc_list))
        fs_error = np.sqrt(np.sum(np.array(fs_list - diff)**2) / len(fs_list))
        print(mc_list,mc_error)
        print(fs_list,fs_error)
        