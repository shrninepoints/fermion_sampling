import numpy as np
import time
import fermion_sampling as fs
from scipy.stats import unitary_group

def local_expectation(temp,N,L,l):
    local_expectation = np.zeros(L//l)
    W = np.zeros((L//l,N,l),dtype='complex')
    for i in range(L // l):
        position = i * l
        W[i] = temp[:,position:(position+l)]
        local_expectation[i] = np.abs(np.trace(np.dot(W[i].T.conj(),W[i])))
    return local_expectation

def sampling2(U,N,L):  # sampling without O(n^2) update method
    v = np.random.permutation(N)
    x = []              # x = [x1,...xk], xi = 0...L-1
    for i in range(N):  # k = i + 1
        p = np.zeros(L)
        for j in range(L):
            submat = U[np.ix_(x+[j],v[0:i+1])]
            p[j] = np.square(np.abs(np.linalg.det(submat)))   # P(x_k = j) 
        prob = p / sum(p) 
        print('xlst = ', x)
        psum = np.zeros(L//l)
        for k in range(L//l):
            position = k * l
            psum[k] = np.sum(prob[position:(position+l)])
        print('psum = ', psum)
        print('expt = ', local_expectation(U[:,v[0:i+1]].T,i+1,L,l) / (i+1))
        x.append(np.random.choice(L,p = prob))          # choose position wrt p  
    m = [1 if i in x else 0 for i in range(L)]          # generate m vector with x
    return np.array(m)

if __name__ == "__main__":
    loop = 1    # number of samples
    N = 4         # num of particle
    L = 256        # length of the system
    U = unitary_group.rvs(L)
    U = U[0:N]
    l = 16
    print(local_expectation(U,N,L,l))
    result = np.zeros(L)
    for i in range(loop):
        result = result + sampling2(U.T,N,L)
    print(result / loop)
    local_number = np.zeros(L//l)
    for i in range(L//l):
        position = i * l
        local_number[i] = np.sum(result[position:(position+l)])
    print(local_number)
    pass