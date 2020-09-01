import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from State import *
from copy import deepcopy
'''
CONST
'''
# system size (2D, sites*L square lattice)
L = 4
N = 16
U = 5         # take tunneling strength as a unit
sites = L**2

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

N_down = np.sum(np.sum(np.abs(phi_0[:L]),0) == 0)
N_up = N - N_down
state = State(sites,np.random.choice(sites, np.int(N_up), replace=False),np.random.choice(sites, np.int(N_down), replace=False))
print(state.getX())

v = 2
epsilon = 0.1
M_optim = 10
Meq = 1000
Mc = 100
interval = sites*2
Mtotal = Meq + Mc * interval

E = np.zeros(M_optim)
V = np.zeros(M_optim+1)
V[0] = v

# loop for gred descent 
for ind_optim in range(M_optim):
    W = np.dot(phi_0, np.linalg.inv(phi_0[state.getX()]))
    mc = 0
    e = 0
    d = 0
    e_mul_d = 0

    np.random.seed()
    for _ in range(Mtotal):
        # randomly choose a new configuration
        state_new = deepcopy(state)
        K,l = state_new.randomHopState()

        # update of Jastrow factor
        J_Ratio = np.exp(-v * (state_new.getDoubleNum() - state.getDoubleNum()))
        
        # Recept in Markov chain
        detRatio = W[K,l]
        r = np.square(np.abs(J_Ratio*detRatio))
        if np.random.rand() < np.min([1, r]):
            state = state_new
            # update of determinant (for details, see below)
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
            e_mul_d = e_mul_d - state.getDoubleNum() * state.energy(v,U,phi_0)
        
    D = d / mc
    E[ind_optim] = e / mc
    E_mul_D = e_mul_d / mc
    f = 2 * (E[ind_optim]*D - E_mul_D)
    v = v + epsilon * f
    V[ind_optim+1] = v

print(V)
print(E)
plt.figure()
plt.plot(E, '+')
plt.figure()
plt.plot(V, '+')
plt.show()
