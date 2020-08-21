import fermion_sampling as fs
import numpy as np

N = 16
size = 4
L = size ** 2
t = 1
u = 2
loop = 1000
v = 1

def genT(N,size,t,u):
    L = size ** 2
    T = np.zeros((2*L,2*L))
    for i in range(L):
        T[i,i+L] = T[i+L,i] = u
        T[i,(i+1) % L] = T[i,(i-1) % L] = T[i,((i+L+1) % L) + L] = T[i,((i+L-1) % L) + L] = -t
        T[i,(i+size) % L] = T[i,(i-size) % L] = T[i,((i+L+size) % L) + L] = T[i,((i+L-size) % L) + L] = -t
        T[i+L,(i+1) % L] = T[i+L,(i-1) % L] = T[i+L,((i+L+1) % L) + L] = T[i+L,((i+L-1) % L) + L] = -t
        T[i+L,(i+size) % L] = T[i+L,(i-size) % L] = T[i+L,((i+L+size) % L) + L] = T[i+L,((i+L-size) % L) + L] = -t
    return T

def calcJ(state,v):
    assert (len(state) == 2*L)
    return np.exp(-1/2 * v * np.dot(state[0:L], state[L:2*L]))

def genU(T):
    e,v = np.linalg.eig(T)
    v = v.T
    U = [x for _,x in sorted(list(zip(e,v)),key=lambda pair: pair[0])]
    U = np.array(U)[0:N]
    return U

def main():
    T = genT(N,size,t,u)
    U = genU(T)
    result = np.zeros(2*L)
    samples = loop
    energy = 0
    normalize = 0
    for _ in range(loop):
        try:
            sample = fs.sampling(U.T,N,2*L)
            J = calcJ(sample,v)
            result = result + J * sample
            energy = energy + J ** 2 * sample.dot(np.dot(T,sample))
            normalize = normalize + J ** 2
        except ValueError:
            samples = samples - 1
    result = result / sum(result) * N
    print(result)
    print(energy / normalize)
    print(samples)
    return result

if __name__ == "__main__":
    state = main()
    pass