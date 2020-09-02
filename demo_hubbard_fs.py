import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from State import *

L = 8       # system size (2D, L*L square lattice)
N = 16      # num of electron
U = 1       # take tunneling strength as a unit
sites = L**2

v = 1      # initial variational parameter
epsilon = 0.1   # variational step length
M_optim = 15    # num of variational steps
loop = 100

E = np.zeros(M_optim)
V = np.zeros(M_optim+1)
V[0] = v

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

H = hamiltonian(L)
e, phi = np.linalg.eigh(H)
phi_0 = phi[: , np.argsort(e)[:N]]

for ind_optim in range(M_optim):
    if ind_optim > 5:
        var = np.var(E[ind_optim-5:ind_optim])
        print(var)
        if var < 0.001: break
        loop = int(1 / var)+100
    samples = loop
    energy = d = e_mul_d = 0
    mat = phi_0 + np.random.rand(sites*2,N) / 10 ** 6     # add small pertubation to avoid singular matrix error
    for _ in range(loop):
        try:
            sample = sampling(mat,N,2*sites)
            sample.sort()
            state = State(sites,sample)
            d = d - state.getDoubleNum()
            energy = energy + state.energy(v,U,phi_0)
            e_mul_d = e_mul_d - state.getDoubleNum() * state.energy(v,U,phi_0)
        except ValueError:
            samples = samples - 1
    if samples / loop < 0.9: raise ValueError("Too many fails")
    D = d / samples
    E[ind_optim] = energy / samples
    E_mul_D = e_mul_d / samples
    f = 2 * (E[ind_optim]*D - E_mul_D)
    v = v + epsilon * f
    V[ind_optim+1] = v

print(E)
print("Energy error = ", np.sqrt(np.var(E[5:])))
print(V)

plt.figure()
plt.plot(E, '+')
plt.figure()
plt.plot(V, '+')
#plt.show()