import sys
import time
from itertools import combinations
import mpmath as mp
mp.mp.dps = 30
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ortho_group
import scipy
from State import *

L = 12
N = 6

def genU(N,L): # generate a random slater determinant to be reconstructed
    U = ortho_group.rvs(L)
    U = U[0:N]
    return U

def det(mat):
    return mp.det(mp.matrix(mat))

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

def constrain(spdict,N,L): # From calc result. return normalized constrain vectors in R^L
    #spdict = extractProb(U,method="part"); print(spdict)
    s = min(spdict, key=spdict.get) # find the most probable state. later all states considered are just one electron differ from s. Probable states have less shot noise
    print(s,spdict[s])
    additional_list = np.random.permutation(np.delete(np.arange(L), s))
    additional_list.sort()
    constrains = np.zeros((L-N,L),dtype=float)
    for i in range(L-N):
        state = list(s) + [additional_list[i]] 
        state.sort()
        state_list = list(combinations(state,N))
        w_list = []
        for ss in state_list:
            ss = tuple(sorted(ss))
            w_list.append(np.sqrt(spdict[ss]))
        state.reverse() # reversed list gives the omitted coordinate in each det. this is related to details in combinations
        for j in range(len(state)):
            constrains[i,state[j]] = w_list[j]
        constrains[i] = constrains[i] / np.sqrt(np.dot(constrains[i],constrains[i]))
        # TODO: assert rank to make sure constrains are linear independent
    return constrains

def constrain2(state_list,weight_list): # From sample result. Using State class
    N = state_list[0].getTotalNum()
    L = state_list[0].sites * 2
    assert N < L/2
    swdict = dict(zip(state_list, weight_list))
    states = list(set(state_list))
    spdict = {tuple(state.getX()): state_list.count(state) * swdict[state] for state in states} # these states are State objects
    s = min(spdict, key=spdict.get) # find the most probable state. later all states considered are just one electron differ from s. Probable states have less shot noise
    additional_list = np.delete(np.arange(L), s)
    constrains = np.zeros((L-N,L),dtype=float)
    for i in range(L-N):
        state = s + [additional_list[i]] # here assumes N < L/2
        state.sort()
        state_list = list(combinations(state,N))
        w_list = []
        for s in state_list:
            s = tuple(sorted(s))
            try :
                w_list.append(np.sqrt(spdict[s]))
            except KeyError: # if not finding this state, assume prob = 0. Maybe improved using e^(-samples) etc.
                w_list.append(0)
        state.reverse() # reversed list gives the omitted coordinate in each det. this is related to details in combinations
        for j in range(len(state)):
            constrains[i,state[j]] = w_list[j]
        constrains[i] = constrains[i] / np.linalg.norm(constrains[i])
    return constrains

def reconstruction(constrains,N,L): # reconsturct with 
    for i in range(N):
        e = np.zeros(L,dtype=np.float); e[i] = 1
        constrains = np.append(constrains,[e],axis=0)
    print(np.linalg.det(constrains))
    basis = [constrains[0]]
    for i in range(1,L): # generate orthonormal basis in R^(L)
        u_next = constrains[i]
        for j in range(len(basis)):            
            u_next = u_next - np.dot(basis[j],constrains[i]) * basis[j]
        u_next = u_next / np.linalg.norm(u_next)
        basis.append(u_next)
    return np.array(basis[L-N:],dtype=np.float)

def hamiltonian(a,b,c,plot=False):
    T = np.zeros((L,L))
    V = np.diag(np.array([a*(-L/2+i*1)**4-b*(-L/2+i*1)**2-c*(-L/2+i*1) for i in range(L)]))
    #plt.figure; plt.plot(np.diag(V)); plt.show()
    if plot: plt.figure(); plt.plot(np.diag(V)); plt.show()
    for i in range(L):
        for j in range(L):
            if i==j: T[i,j] = (np.pi/1)**2 / 6
            else: T[i,j] = (-1) ** (i-j) / ((i-j)**2 * 1 ** 2)
    return T+V

if __name__ == "__main__":
    H = hamiltonian(0,0,0)
    e, phi = np.linalg.eigh(H)
    phi_0 = phi[: , np.argsort(e)[:N]]
    U = phi_0[:,:N].T #genU(N,L)
    #print(U)
    sp1 = extractProb(U)
    #print(sp1)
    constrains = constrain(sp1,N,L)
    reU = reconstruction(constrains,N,L)
    #print(reU)
    sp2 = extractProb(reU)
    #print(sp2)
    p1 = list(sp1.values()); p2 = list(sp2.values()) 
    print(np.dot(p1,p2)/(np.linalg.norm(p1) * np.linalg.norm(p2)))
    
    pass
