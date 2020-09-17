import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.stats import ortho_group
from itertools import combinations 

N = 2
L = 4

def genU(N,L): # generate a random slater determinant to be reconstructed
    U = ortho_group.rvs(L)
    U = U[0:N]
    return U

def calcW(U,x): # calc determinant of the state, with input x a N-d list
    return np.linalg.det(U[x])

def extractProb(U,num = None,method = "all"): # extract num differet states and their probability, return two lists (prob_list,state_list)
    N,L = np.shape(U)
    prob_list = []
    if method == "part": # produce random states, for case num << all state
        pass # TODO: complete for high dimension case
    elif method == "all": # produce all states
        state_list = list(combinations([_ for _ in range(L)], N))
        if num is not None: state_list = state_list[:num]
        for state in state_list:
            prob_list.append(np.square(np.linalg.det(U[:,list(state)])))
    return dict(zip(state_list, prob_list)) # states are already sorted

def constrain(U,N,L): # return normalized constrain vectors in R^L
    spdict = extractProb(U)
    constrains = np.zeros((L-N,L),dtype=float)
    prev_state_list = []
    it = iter(spdict)
    for i in range(L-N):
        while True:
            state = list(next(it)) # return a key in spdict
            additional = np.random.choice(np.delete(np.arange(L), state))
            state.append(additional)
            state.sort()
            if state not in prev_state_list: break
        prev_state_list.append(state.copy())
        state_list = list(combinations(state,N))
        w_list = []
        for s in state_list:
            s = tuple(sorted(s))
            w_list.append(np.sqrt(spdict[s]))
        #del spdict[tuple(state[0:-1])]
        state.reverse() # reversed list gives the omitted coordinate in each det. this is related to details in combinations
        for j in range(len(state)):
            constrains[i,state[j]] = w_list[j]
        constrains[i] = constrains[i] / np.linalg.norm(constrains[i])
        # assert rank to make sure constrains are linear independent
    return constrains

def reconstruction(constrains,N,L): # reconsturct with 
    basis = [constrains[0]]
    for i in range(1,len(constrains)):
        u_next = constrains[i]
        for j in range(len(basis)):            
            u_next = u_next - np.dot(basis[j],constrains[i]) * basis[j]
        u_next = u_next / np.linalg.norm(u_next)
        basis.append(u_next)
    print(basis)
    for i in range(N):
        e = np.zeros(L,dtype=float); e[i] = 1
        tmp = np.dot(basis,e)
        for j in range(len(tmp)):
            e = e - tmp[j] * basis[j]
        assert np.linalg.norm(e) > 10 ** -3
        e = e / np.linalg.norm(e)
        basis = np.append(basis,[e],axis=0)
    return basis[L-N:]


if __name__ == "__main__":
    U = genU(N,L)
    print(U)
    print(extractProb(U))
    constrains = constrain(U,N,L)
    #print(constrains)
    reU = reconstruction(constrains,N,L)
    print(extractProb(reU))
    print(np.dot(reU[0],reU[1]))
    pass