import sys
from itertools import combinations, product
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import block_diag
import demo_doublewell
from State1d import *

L = 9
N = 9
v = 0.4
a = 2 
b = 8
c = 1.7
xrange = 2
interval = xrange / (L-1) * 2

def extractProb(U,num = None,method = "all",probable_state = None): # extract num differet states and their probability, return two lists (prob_list,state_list)
    N,L = np.shape(U)
    prob_list = []
    state_list = []
    if method == "part": # produce random states, for case num << all state
        if probable_state is not None: 
            random_state = probable_state
        else: 
            random_state = list(np.random.choice(L, N, replace=False))
        additional_list = np.random.permutation(np.delete(np.arange(L), random_state))
        for i in range(L-N):
            state = random_state + [additional_list[i]] 
            state.sort()
            sl = list(combinations(state,N))
            state_list = state_list + sl
            for state in sl:
                prob_list.append(np.square(np.linalg.det(U[:,list(state)])))
    elif method == "all": # produce all states
        state_list = list(combinations([_ for _ in range(L)], N))
        if num is not None: state_list = state_list[:num]
        for state in state_list:
            prob_list.append(np.square(np.linalg.det(U[:,list(state)])))
    return dict(zip(state_list, prob_list)) # states are already sorted

def extractWeightedProb(U,v,N,L,method = "all"): # spin half, 1d case
    N_down = np.sum(np.sum(np.abs(U.T[:L//2]),0) == 0)
    N_up = N - N_down
    prob_list = []
    state_list = []
    if method == "part":
        pass
    elif method == "all": # produce all states
        #state_list = list(combinations([_ for _ in range(L)], N))
        state_list_up = list(combinations([_ for _ in range(L)],N_up))
        state_list_down = list(combinations([_ for _ in range(L)],N_down))
        state_list = list(product(state_list_up,state_list_down))
        for i in range(len(state_list)):
            state = State1d(L,state_list[i][0],state_list[i][1])
            p = state.getWeight(v) * np.square(np.linalg.det(U[:,state.getX()]))
            prob_list.append(p)
            state_list[i] = tuple(state.getX())
    print(np.sum(prob_list))
    return dict(zip(state_list, prob_list)) # states are already sorted


def constrain(spdict,N,L): # From calc result. return normalized constrain vectors in R^L
    s = min(spdict, key=spdict.get) # find the most probable state. later all states considered are just one electron differ from s. Probable states have less shot noise
    additional_list = np.random.permutation(list(np.delete(np.arange(L), s[:N//2])) + list(np.delete(np.arange(2*L),s[N//2:]))[L:])
    additional_list.sort()
    constrains = np.zeros((2*L-N,2*L),dtype=float)
    for i in range(2*L-N):
        state = list(s) + [additional_list[i]] 
        state.sort()
        state_list = list(combinations(state,N))
        w_list = []
        for ss in state_list:
            ss = tuple(sorted(ss))
            try:
                w_list.append(np.sqrt(spdict[ss]))
            except KeyError:
                w_list.append(0)
        state.reverse() # reversed list gives the omitted coordinate in each det. this is related to details in combinations
        for j in range(len(state)):
            constrains[i,state[j]] = w_list[j]
        constrains[i] = constrains[i] / np.sqrt(np.dot(constrains[i],constrains[i]))
        # TODO: assert rank to make sure constrains are linear independent
    return constrains

def reconstructionSpin(constrains,N,L):
    def reconstruction(constrains,N,L): # reconsturct with 
        for i in range(N):
            e = np.zeros(L); e[i] = 1
            constrains = np.append(constrains,[e],axis=0)
        basis = [constrains[0]]
        for i in range(1,L): # generate orthonormal basis in R^(L)
            u_next = constrains[i]
            for j in range(len(basis)):            
                u_next = u_next - np.dot(basis[j],constrains[i]) * basis[j]
            u_next = u_next / np.linalg.norm(u_next)
            basis.append(u_next)
        return np.array(basis[L-N:])
    tmp = reconstruction(constrains[:(2*L - N) // 2,:L],N//2,L) # don't need to generate the whole R^(2L) space. Just two R^(L) space
    tmp2 = np.flip(tmp,0)
    tmp2 = np.flip(tmp2,1)
    reU = block_diag(tmp,tmp2)
    return reU

def hamiltonian(a,b,c,plot=False, discrete=False):
    T = np.zeros((2*L,2*L))
    V_diag = np.array([a*(-xrange+i*interval)**4-b*(-xrange+i*interval)**2-c*(-xrange+i*interval) for i in range(L)])
    V = np.diag(np.append(V_diag, V_diag))
    # print('c={}, V_diag:{}'.format(c, V_diag))
    if plot: plt.figure(); plt.plot(V_diag); plt.show()
    if discrete:
        for i in range(2*L-1):
            T[i, i+1] = -1
        T[L-1, L] = 0
        T = T + T.T
    else:
        for i in range(L):
            for j in range(L):
                if i==j: T[i,j] = T[i+L,j+L] = (np.pi/interval)**2 / 6
                else: T[i,j] = T[i+L,j+L] = (-1) ** (i-j) / ((i-j)**2 * interval ** 2)
    return T+V

def preprocess(U,v,N,L):
    sp = extractWeightedProb(U,v,N,L)
    constrains = constrain(sp,N,L)
    reU = reconstructionSpin(constrains,N,L)
    return reU

if __name__ == "__main__":
    H = hamiltonian(0,0,0,discrete=True)
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    U = phi_0.T #genU(N,L)
    print(U)

    assert np.sum(np.sum(np.abs(U.T[:L]),0) == 0) == N // 2 # spin up == spin down
    sp1 = extractWeightedProb(U,v,N,L)
    #print(sp1)
    constrains = constrain(sp1,N,L)
    reU = reconstructionSpin(constrains,N,L)
    print(reU) # reconstructed slater determinant
    sp2 = extractWeightedProb(reU,0,N,L)
    #print(sp2)
    p1 = list(sp1.values()); p2 = list(sp2.values()) 
    print(np.dot(p1,p2)/(np.linalg.norm(p1) * np.linalg.norm(p2)))

    pass
