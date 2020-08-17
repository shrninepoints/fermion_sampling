import numpy as np
import time
import fermion_sampling as fs

def sampleU(U,N,L,e):
    U_temp = []
    for i in range(N):
        if np.random.rand() < (e[i]/(e[i]+1)):
            U_temp.append(list(U[i]))
    result = fs.dpp(np.array(U_temp),len(U_temp),L)
    return result       

def bit2int(m):
        m_int = 0                           
        for i in m:
            m_int = (m_int << 1) | i
        return m_int

if __name__ == "__main__":
    N = 4
    L = 5
    loop = 1
    B = np.zeros((N,L))
    B0 = np.array([i/L for i in range(L)])
    for i in range(N):
        B[i] = np.power(B0,i)
        B[i] = B[i] / np.sqrt(np.dot(B[i],B[i]))
    l = np.dot(B.T,B)
    K = l.dot(np.linalg.inv(l+np.identity(L)))
    e,v = np.linalg.eig(l)
    U = v.T[0:N]
    result = []
    for _ in range(loop):
        result.append(bit2int(sampleU(U,N,L,e)))
    count = [result.count(i)/loop for i in range(2 ** L)]
    print(count)
    fs.sample_state(loop,'sampling',U,N,L)
    fs.exact(U,N,L)
    pass