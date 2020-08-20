'''
potential optimisation: using GPU,
'''
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import expm
import time
import matplotlib.pyplot as plt

def dpp(U,N,L):
    K = U
    #K = np.dot(U.T.conj(),U)
    p = np.zeros(L)
    for i in range(L):
        v = K[:,i]
        p[i] = np.square(np.linalg.norm(v))
    prob = p / sum(p)   
    Xn = np.random.choice(L,p = prob) # sample Xn
    e = np.asmatrix(K[:,Xn] / np.linalg.norm(K[:,Xn])) # set e1
    X = [0 for i in range(N)]
    X[N-1] = Xn
    for i in range(N-1,0,-1):
        p = np.zeros(L)
        for j in range(L):
            v = K[:,j]
            p[j] = (np.square(np.linalg.norm(v)) - \
                np.sum(np.square(np.abs(np.dot(e.conj(),v)))))   # P_i = i(x = j)
        prob = abs(p) / sum(abs(p))    
        X[i-1] = np.random.choice(L,p = prob)
        vXi = K[:,X[i-1]]
        wi = vXi
        for j in range(N-i):
            wi = wi - np.dot(e[j,:].conj(),vXi) * e[j,:]
        e_next = wi / np.linalg.norm(wi)
        e = np.append(e,e_next,axis = 0)
    m = [1 if i in X else 0 for i in range(L)]          # generate m vector with x
    return np.array(m)


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
    m = [1 if i in x else 0 for i in range(L)]          # generate m vector with x
    return np.array(m)

def sampling1(U,N,L):
    v = np.random.permutation(N)
    x = []              # x = [x1,...xk], xi = 0...L-1
    for i in range(N):  # k = i + 1
        p = np.zeros(L,dtype='float128')
        first_matrix = None
        for j in range(L):
            if j not in x: 
                if first_matrix is None:
                    # TODO: pivoting, if first matrix is too small, choose another one
                    j0 = j
                    first_matrix = U[np.ix_(x+[j],v[0:i+1])] # TODO: dealing with matrix element = 0
                    first_matrix_inv = np.linalg.inv(first_matrix)
                    first_matrix_det = np.linalg.det(first_matrix)
                    p[j] = np.square(np.abs(first_matrix_det))
                else:
                    delta_v = U[np.ix_([j],v[0:i+1])] - U[np.ix_([j0],v[0:i+1])]
                    detRatio = 1 + np.dot(first_matrix_inv[:,-1],delta_v[0,:])
                    p[j] = np.square(np.float128(np.abs(detRatio * first_matrix_det)))
                    # use of higher accuracy is necessary, otherwise introduce divide by 0 for small first_matrix_det
        prob = np.float64(p / sum(p))
        x.append(np.random.choice(L,p = prob))          # choose position wrt p  
    m = [1 if i in x else 0 for i in range(L)]          # generate m vector with x         # generate m vector with x
    return np.array(m) 

def sampling2(U,N,L):  # sampling without O(n^2) update method
    v = np.random.permutation(N)
    x = []              # x = [x1,...xk], xi = 0...L-1
    for i in range(N):  # k = i + 1
        p = np.zeros(L)
        for j in range(L):
            submat = U[np.ix_(x+[j],v[0:i+1])]
            p[j] = np.square(np.abs(np.linalg.det(submat)))   # P(x_k = j) 
        prob = p / sum(p) 
        x.append(np.random.choice(L,p = prob))          # choose position wrt p  
    m = [1 if i in x else 0 for i in range(L)]          # generate m vector with x
    return np.array(m)

def exact(phi,N,L):  # exact solution for system with length < 8
    statelist = np.array([i for i in range(2 ** L)],dtype='uint8')[np.newaxis]
    statelist = np.unpackbits(statelist.T, axis=1) # unpack int to bit list, i.e. states
    statelist = np.array([x for x in statelist if np.count_nonzero(x==1) == N]) # remove states with number of 1 != N
    statelist = statelist[:,8-L:8] # remove 0 in the head
    p = np.zeros(len(statelist))
    for i in range(len(statelist)):
        phi_state = np.array([phi[:,j] for j in range(L) if statelist[i,j] == 1])
        p[i] = np.square(np.abs(np.linalg.det(phi_state)))
    prob = p / sum(p)
    print(prob)
    return prob

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args[0:2], kw, te-ts))
        return result
    return timed

@timeit
def sample_position(loop,method,U,N,L):
    result = np.zeros(L)
    if method == "dpp":
        for _ in range(loop):
            result = result + dpp(U,N,L)
    elif method == "sampling":
        for _ in range(loop):
            result = result + sampling(U.T,N,L)
    elif method == "sampling1":
        for _ in range(loop):
            result = result + sampling1(U.T,N,L)
    elif method == "sampling2":
        for _ in range(loop):
            result = result + sampling2(U.T,N,L)
    else:
        raise Exception("UnknownMethod")
    print(result / loop)
    return result / loop
    
@timeit
def sample_state(loop,method,U,N,L):
    def bit2int(m):
        m_int = 0                           
        for i in m:
            m_int = (m_int << 1) | i
        return m_int
    result = []
    if method == "dpp":
        for _ in range(loop):
            result.append(bit2int(dpp(U,N,L)))
    elif method == "sampling":
        for _ in range(loop):
            result.append(bit2int(sampling(U.T,N,L)))
    elif method == "sampling1":
        for _ in range(loop):
            result.append(bit2int(sampling1(U.T,N,L)))
    elif method == "sampling2":
        for _ in range(loop):
            result.append(bit2int(sampling2(U.T,N,L)))
    elif method == "exact":
        for _ in range(loop):
            exact(U,N,L)
    else:
        raise Exception("UnknownMethod")
    count = [result.count(i)/loop for i in range(2 ** L)]
    print(count)
    return count

if __name__ == "__main__":
    loop = 10000    # number of samples
    N = 4         # num of particle
    L = 16        # length of the system
    U = unitary_group.rvs(L)
    U = U[0:N,:]
    '''
    B = np.zeros((N,L))
    B0 = np.array([i/L for i in range(L)])
    for i in range(N):
        B[i] = np.power(B0,i)
        B[i] = B[i] / np.sqrt(np.dot(B[i],B[i]))
    l = np.dot(B.T,B)
    K = l.dot(np.linalg.inv(l+np.identity(L)))
    e,v = np.linalg.eig(l)
    U = v.T[0:N]
    '''
    print(U)
    
    sample_position(loop,"sampling2",U,N,L)
    sample_position(loop,"sampling",U,N,L)
    '''
    count0 = np.array(sample_state(loop,"dpp",U,N,L))
    count0 = count0[count0 != 0]    
    count1 = np.array(sample_state(loop,"sampling2",U,N,L))
    count1 = count1[count1 != 0]
    count2 = np.array(exact(U,N,L))
    count2 = count2[count2 != 0]
    
    v0_u = count0 / np.linalg.norm(count0)
    v1_u = count1 / np.linalg.norm(count1)
    v2_u = count2 / np.linalg.norm(count2)
    print("theta01 = ",np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))) # dpp - sampling
    print("theta02 = ",np.arccos(np.clip(np.dot(v0_u, v2_u), -1.0, 1.0))) # dpp - exact
    print("theta12 = ",np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))) # sampling -exact '''
    pass