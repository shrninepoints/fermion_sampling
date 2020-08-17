import numpy as np
import time
import fermion_sampling as fs

def hamiltonian_3d(w,g,size):
    l = size ** 3
    e = (np.random.rand(l) - 0.5) * w
    h = np.diag(e)
    for i in range(l):
        h[i,(i+1) % l] = h[i,i-1] = -g
        h[i,(i+size) % l] = h[i,i-size] = -g
        h[i,(i+size ** 2) % l] = h[i,i-size ** 2] = -g
    return h

def slater(h):
    e, v = np.linalg.eig(h)
    v = v.T
    U = [x for _,x in sorted(list(zip(e,v)),key=lambda pair: pair[0])]
    E = [y for y,_ in sorted(list(zip(e,v)),key=lambda pair: pair[0])]
    print(E)
    energy = E[N-1]
    U = np.array(U)[0:N]
    return energy, U

if __name__ == "__main__":
    w = 20
    g = 1
    size = 8  
    N_list = [8]
    print('w = %s, g = %s, size = %s, N_list = %s' % (w,g,size,N_list))
    loop = 100
    for N in N_list: 
        ts = time.time()
        l = []
        fail = 0
        samples = 1
        fermi_energy = 0
        while len(l) < samples:
            try:
                energy,U = slater(hamiltonian_3d(w,g,size))
                fermi_energy = fermi_energy + energy
                result = np.zeros(size ** 3)
                for _ in range(loop):
                        result = result + fs.sampling(U.T,N,size ** 3)
                half_particle_num = sum(result[0:(size ** 3)//2] / loop)
                l.append(half_particle_num - N//2)
            except: # for very werid value, divid by 0 error may occur. just ignore these results
                fail = fail + 1
        fermi_energy = fermi_energy / (samples - fail)
        print('fermi energy = ',fermi_energy)
        print(l)
        print('variance = ',np.var(l),'with N = ',N)
        print('total_fail = ',fail)
        te = time.time()
        print('samples:%s loop:%s took: %2.4f sec' % (samples, loop, te-ts))
    pass