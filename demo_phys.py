import numpy as np
import time
import fermion_sampling as fs

L = 32
N = 16
disorder = 2

def hamiltonian_3d(w,g,size):
    l = size ** 3
    e = (np.random.rand(l) - 0.5) * w
    h = np.diag(e)
    for i in range(l):
        h[i,(i+1) % l] = h[i,i-1] = -g
        h[i,(i+size) % l] = h[i,i-size] = -g
        h[i,(i+size ** 2) % l] = h[i,i-size ** 2] = -g
    return h

def hamiltonian_1d(disorder):
    H1 = np.zeros((L,L))
    for i in range(L):
        H1[i, (i+1) % L] = -1
    H = H1 + H1.T
    np.random.seed()
    impurity = (np.random.rand(L) - 0.5) * 2
    H = H1 + disorder * np.diag(impurity)
    H = H + np.random.rand(L,L) / 10 ** 8
    return H

def slater(h):

    return energy, U

if __name__ == "__main__":
    H = hamiltonian_1d(disorder)
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]