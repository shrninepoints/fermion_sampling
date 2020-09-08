import numpy as np

class State:
    def __init__(self,*args): # initialize with (x_up,x_down) or x
        self.sites = args[0]
        self.up = args[1]
        self.down = args[2]
    
    def getUpNum(self):
        return len(self.up)
    def getDownNum(self):
        return len(self.down)
    def getTotalNum(self):
        return len(self.up) + len(self.down)
    def getX(self):
        return np.append(self.up, self.down+self.sites)
    def getDoubleOccupance(self):
        return np.intersect1d(self.up, self.down)
    def getDoubleNum(self):
        return len(self.getDoubleOccupance())
    def getEmptyUp(self):
        return np.delete(np.arange(self.sites), self.up)
    def getEmptyDown(self):
        return np.delete(np.arange(self.sites), self.down)
    def getL(self):
        return int(np.sqrt(self.sites))
    def getState(self):
        m = [1 if i in self.getX() else 0 for i in range(self.sites * 2)]
        return m


    def onehop(self):
        def hop(x_in):
            result_hop = []
            # left hop
            for i in range(len(x_in)):
                if x_in[i] % L == 0:
                    if x_in[i]+L-1 in x_in:
                        continue
                    else:
                        left = x_in.copy()
                        left[i] = x_in[i]+L-1
                        result_hop.append(left)    
                elif x_in[i]-1 in x_in:
                    continue
                else:
                    left = x_in.copy()
                    left[i] = left[i] - 1
                    result_hop.append(left)
            # right hop
            for i in range(len(x_in)):
                if x_in[i] % L == L-1:
                    if x_in[i]+1-L in x_in:
                        continue
                    else:
                        right = x_in.copy()
                        right[i] = x_in[i]+1-L
                        result_hop.append(right)
                elif x_in[i]+1 in x_in:
                    continue
                else:
                    right = x_in.copy()
                    right[i] = right[i] + 1
                    result_hop.append(right)
            # up hop
            for i in range(len(x_in)):
                if x_in[i] in range(L):
                    if x_in[i]-L+self.sites in x_in:
                        continue
                    else:
                        up = x_in.copy()
                        up[i] = x_in[i]-L+self.sites
                        result_hop.append(up)    
                elif x_in[i]-L in x_in:
                    continue
                else:
                    up = x_in.copy()
                    up[i] = up[i] - L
                    result_hop.append(up)
            # down hop
            for i in range(len(x_in)):
                if self.sites-1-x_in[i] in range(L):
                    if x_in[i]+L-self.sites in x_in:
                        continue
                    else:
                        down = x_in.copy()
                        down[i] = x_in[i]+L-self.sites
                        result_hop.append(down)    
                elif x_in[i]+L in x_in:
                    continue
                else:
                    down = x_in.copy()
                    down[i] = down[i] + L
                    result_hop.append(down)
            return result_hop
        L = self.getL()
        hop_up = hop(self.up)
        hop_down = hop(self.down)
        hop_up = [State(self.sites,item,self.down) for item in hop_up]
        hop_down = [State(self.sites,self.up,item) for item in hop_down]
        return hop_up + hop_down 

    def energy(self,v,U,phi_0):
        psi_x = np.linalg.det(phi_0[self.getX()])
        assert psi_x != 0
        jx = np.exp(-v * self.getDoubleNum())
        psi_jx = psi_x * jx
        ux = U * self.getDoubleNum()
        x_prime = self.onehop()
        double_prime = np.array([state.getDoubleNum() for state in x_prime])
        psi_x_prime = np.array([np.linalg.det(phi_0[state.getX()]) for state in x_prime])
        jx_prime = np.exp(-v * double_prime)        
        E0 = (-np.sum(jx_prime * psi_x_prime) + ux*psi_jx) / psi_jx
        return E0

    @staticmethod
    def distance2D(L,a,b):
        def distance1D(a,b):
            d = np.abs(a-b)
            return d if d <= np.int(L/2) else L - d
        a_coor = (a % L,a // L)
        b_coor = (b % L,b // L)
        return distance1D(a_coor[0],b_coor[0]) + distance1D(a_coor[1],b_coor[1])

    # for a given configuration x, return the value of diag elements of local operator O_k(x)
    # distance k ranges from 0 to int(L/2) 
    def O_k(self):
        o = np.zeros(self.getL() // 2 * 2 + 1,dtype=np.int8)
        site_double = np.intersect1d(self.up, self.down)
        site_occ = np.union1d(self.up, self.down)
        n = np.zeros(self.sites, dtype=np.int8)
        n[site_occ] = 1
        n[site_double] = 2
        o[0] = np.dot(n, n)
        for ind in range(len(site_occ)):
            i = site_occ[ind]
            n_i = n[i]
            for ind1 in range(len(site_occ)-ind-1):
                j = site_occ[ind+ind1+1]
                n_j = n[j]
                k = State.distance2D(self.getL(),i,j)
                o[k] = o[k] + n_i * n_j       
        return o

    def energy_jastrow(self,v,U,phi_0):
        psi_x = np.linalg.det(phi_0[self.getX()])
        assert psi_x != 0
        jx = np.exp(-np.dot(v, self.O_k()))
        psi_jx = psi_x * jx
        ux = U * self.getDoubleNum()
        x_prime = self.onehop()
        psi_x_prime = np.array([np.linalg.det(phi_0[state.getX()]) for state in x_prime])
        jx_prime = np.exp(-np.array([np.dot(v, state.O_k()) for state in x_prime]))
        E0 = (-np.sum(jx_prime * psi_x_prime) + ux*psi_jx) / psi_jx
        return E0

    def randomHopState(self):
        index_change = np.random.randint(self.getTotalNum())
        if index_change < self.getUpNum():
            index_new = np.random.randint(self.sites - self.getUpNum())
            self.up[index_change] = self.getEmptyUp()[index_new]
            K = self.up[index_change]
        else:
            index_new = np.random.randint(self.sites - self.getDownNum())
            self.down[index_change-self.getUpNum()] = self.getEmptyDown()[index_new]
            K = self.down[index_change-self.getUpNum()] + self.sites
        return (K,index_change)
