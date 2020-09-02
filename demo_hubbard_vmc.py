import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from State import *
from copy import deepcopy

L = 3       # system size (2D, L*L square lattice)
N = 6      # num of electron
U = 1       # take tunneling strength as a unit
sites = L**2

v = 0.5      # initial variational parameter
epsilon = 0.1   # variational step length
M_optim = 20    # num of variational steps
Meq = 1000      # vmc step to reach near ground state
Mc = 100        # vmc step to accumulate observable
interval = sites*2
Mtotal = Meq + Mc * interval

E = np.zeros(M_optim)
V = np.zeros(M_optim+1)
V[0] = v

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
state = State(sites,np.random.choice(sites, np.int(N_up), replace=False),np.random.choice(sites, np.int(N_down), replace=False))    # initial state

# loop for grad descent 
for ind_optim in range(M_optim):
    W = np.dot(phi_0, np.linalg.inv(phi_0[state.getX()]))
    mc = e = d = e_mul_d = 0
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
            d = d - state.getDoubleNum()
            e = e + state.energy(v,U,phi_0)
            e_mul_d = e_mul_d - state.getDoubleNum() * state.energy(v,U,phi_0)

    D = d / mc
    E[ind_optim] = e / mc
    E_mul_D = e_mul_d / mc
    f = 2 * (E[ind_optim]*D - E_mul_D)
    v = v + epsilon * f
    V[ind_optim+1] = v

print(E)
print(V)

plt.figure()
plt.plot(E, '+')
plt.figure()
plt.plot(V, '+')
plt.show()