'''
1d double well potential
'''
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 1, 2**8 -> 2.055419921875, 2,2**8 -> 1.96923828125
b = 2 ** 8
a = b / 2 ** 3
c = 11.31463405 # V = ax^4 - bx^2 - cx  #12,48 -> 5.093, 24,96 -> 6.978545, 96,384 -> 13.856471078 2**16,2**18->738.8916015625 1.3366
xrange = 2 # position takes value from [-xrange,xrange]
L = 32
N = 16
interval = xrange / (L-1) * 2
s = 3
p = True
disorder = 5

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args[0:2], kw, te-ts))
        return result
    return timed

def hamiltonian(a,b,c,plot=False):
    T = np.zeros((L,L))
    V = np.diag(np.array([a*(-xrange+i*interval)**4-b*(-xrange+i*interval)**2-c*(-xrange+i*interval) for i in range(L)]))
    if plot: plt.figure(); plt.plot(np.diag(V)); plt.show()
    for i in range(L):
        for j in range(L):
            if i==j: T[i,j] = (np.pi/interval)**2 / 6
            else: T[i,j] = (-1) ** (i-j) / ((i-j)**2 * interval ** 2)
    return T,V


def hamiltonian_anderson(disorder):
    H1 = np.zeros((L,L))
    for i in range(L):
        H1[i, (i+1) % L] = -1
    H = H1 + H1.T
    np.random.seed()
    impurity = (np.random.rand(L) - 0.5) * 2
    H = H1 + disorder * np.diag(impurity)
    H = H + np.random.rand(L,L) / 10 ** 8
    return H

def h2diff(tv):
    H = tv[0] + tv[1]
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    position = [np.sum(phi_0[i] ** 2) for i in range(L)]
    diff = N - 2*np.sum(position[:L//2])
    return diff

def search(a,b):
    m = 1024
    n = 0
    recursion = 0
    while True:
        recursion += 1
        diff = h2diff(hamiltonian(a,b,(m+n)/2,plot=False))
        if diff < 0.5: n = (m+n)/2
        elif diff > 1.5: m = (m+n)/2
        else: break
        if recursion > 512: print("fail with ",a,b,"end with ",(m+n)/2); break
    return (m+n)/2

def searchAll(b):
    def sharpness(a,b,c):
        m = h2diff(hamiltonian(a,b,c))
        n = h2diff(hamiltonian(a,b,c + 10 ** -8))
        return (n-m)*10**8
    sharpness_list = []
    points = 32
    arange = [0,b]
    for i in range(points):
        a = arange[0] + i * (arange[1] - arange[0]) / points
        c = search(a,b)
        sharpness_list.append(sharpness(a,b,c))
    plt.figure(); plt.plot(sharpness_list); plt.show()
    pass


def corr_length(c):
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

        def correlation_length(e_list):
            def autocorrelation(step,e_list):
                assert len(e_list[step:]) == len(e_list[0:len(e_list)-step])
                cov = np.cov(e_list[step:],e_list[0:len(e_list)-step])[0][1]
                return cov / np.var(e_list)
            ac_list = []
            for step in range(900):
                ac_list.append(autocorrelation(step,e_list))
            try:
                tau = abs(np.polyfit(np.log(np.abs(ac_list[0:20])), np.arange(20), 1)[0])
            except:
                tau = 0
            print("tau = ",tau) # corr length / (sites*2)
            print("integrated tau = ", np.sum(ac_list))        
            #plt.figure(); plt.plot(ac_list); plt.show()   
            return tau

        state = np.random.choice(L, N, replace=False)
        W = np.dot(phi_0, np.linalg.inv(phi_0[state]))
        mc = e = diff = 0
        position = np.zeros(L)
        np.random.seed()
        Mtotal = Meq + Mc * N
        diff_list = []
        pot_list = []
        for _ in range(Mtotal):
            # randomly choose a new configuration
            state_new = state.copy()
            K,l = randomHopState(state_new)
            detRatio = W[K,l] # Recept in Markov chain
            r = np.square(np.abs(detRatio))
            #print(r)
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
                #print(state)
                position += np.array([1 if i in state else 0 for i in range(L)])
                diff += N - 2*sum(x < L/2 for x in state)
                diff_list.append(N - 2*sum(x < L/2 for x in state))
                #m = np.array([1 if i in state else 0 for i in range(L)])
                #potential = np.dot(m,np.diag(V))
                #pot_list.append(potential)
        diff = diff / mc
        position = position/mc
        tau = correlation_length(diff_list)
        #tau_p = correlation_length(pot_list)
        #plt.figure(); plt.plot(position); plt.show()
        return tau
    
    file = open('data/'+str(c)+'out.txt', 'w')
    sys.stdout = file
    print(a,b,c,xrange,L,N)
    #T,V = hamiltonian(a,b,c,plot=False); H = T + V
    H = hamiltonian_anderson(disorder)
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    position = [np.sum(phi_0[i] ** 2) for i in range(L)]
    #plt.figure(); plt.plot(position); plt.show()
    diff = N - 2*np.sum(position[:L//2])
    print("theo diff = ", diff)
    #print("theo pot = ",np.dot(position,np.diag(V)))
    mc_list = []
    for i in range(5):
        mc_list.append(mc(10000,10000))
        #fs_list.append(fs(100))
    mc_list = [i for i in mc_list if i != 0]
    mc_error = np.sqrt(np.sum(np.array(mc_list - np.sum(mc_list)/len(mc_list))**2) / len(mc_list))
    #fs_error = np.sqrt(np.sum(np.array(fs_list - diff)**2) / len(fs_list))
    print(mc_list,mc_error,np.sum(mc_list)/len(mc_list))
    #print(fs_list,fs_error)
    file.close()
    return np.sum(mc_list)/len(mc_list)

if __name__ == "__main__":
    if s == 1:
        #searchAll(b)
        T,V = hamiltonian(a,b,c,plot=p)
        print(search(a,b))
        print(h2diff(hamiltonian(a,b,c)))
    elif s == 0:
        l = [60.66064 + i * 0.25 * 10 ** -5 for i in range(24)]
        with Pool(8) as p:
            tau_list = p.map(corr_length, [c])
        for tau in tau_list:
            print(tau)
    elif s ==3:
        print(disorder)
        corr_length(0)
    '''
    a = 2 ** -6
    b = 2 ** 3
    c = 0.4449462890625 # V = ax^4 - bx^2 - cx  #12,48 -> 5.093, 24,96 -> 6.978545, 96,384 -> 13.856471078 2**16,2**18->738.8916015625 1.3366
    xrange = 32 # position takes value from [-xrange,xrange]
    L = 64
    N = 4   
    interval = xrange / (L-1) * 2
    s = 2
    T,V = hamiltonian(a,b,c,plot=True)        
    H = T+V
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    position = [np.sum(phi_0[i] ** 2) for i in range(L)]
    diff = N - 2*np.sum(position[:L//2])
    print("theo diff = ", diff)
    e.sort(); print(e[0:N])
    print(search(a,b))
    for i in range(N):
        plt.figure(); plt.plot(phi_0[:,i]);plt.show()
    '''

