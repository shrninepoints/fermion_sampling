import numpy as np
from State import *

class State1d(State):

    def onehop(self, pb=True):
        def hop(x_in):
            result_hop = []
            # left hop
            for i in range(len(x_in)):
                if x_in[i] % L == 0:
                    if pb:
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
                    if pb:
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
            
            return result_hop

        L = self.sites
        hop_up = hop(self.up)
        hop_down = hop(self.down)
        hop_up = [State1D(self.sites,item,self.down) for item in hop_up]
        hop_down = [State1D(self.sites,self.up,item) for item in hop_down]
        return hop_up + hop_down

    def energy(self,v,U,phi_0,pb=True):
        psi_x = np.linalg.det(phi_0[self.getX()])
        assert psi_x != 0
        jx = np.exp(-v * self.getDoubleNum())
        psi_jx = psi_x * jx
        ux = U * self.getDoubleNum()
        x_prime = self.onehop(pb=pb)
        double_prime = np.array([state.getDoubleNum() for state in x_prime])
        psi_x_prime = np.array([np.linalg.det(phi_0[state.getX()]) for state in x_prime])
        jx_prime = np.exp(-v * double_prime)        
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

    def correlation_c(self,v,phi_0):
        x = self.getX()
        psi_0_x = np.linalg.det(phi_0[self.getX()])
        assert psi_0_x != 0
        jx = np.exp(-v * self.getDoubleNum())
        psi_jx = jx * psi_0_x
        c_up = np.zeros((self.sites, self.sites))
        c_down = np.zeros((self.sites, self.sites))

        for ind in range(self.getUpNum()):
            row_i = np.zeros(self.sites)
            for j in range(self.sites):
                x_hop = x.copy()
                x_hop[ind] = j
                x_up_hop = (self.up).copy()
                x_up_hop[ind] = j
                double_hop = np.size(np.intersect1d(x_up_hop, self.down))
                jx_hop = np.exp(-v * double_hop)
                row_i[j] = jx_hop * np.linalg.det(phi_0[x_hop])
            c_up[x[ind]] = row_i
        
        for ind_down in range(self.getDownNum()):
            row_i = np.zeros(self.sites)
            for j in range(self.sites):
                x_hop = x.copy()
                x_hop[ind_down+self.getUpNum()] = j+self.sites
                x_down_hop = (self.down).copy()
                x_down_hop[ind_down] = j
                double_hop = np.size(np.intersect1d(self.up, x_down_hop))
                jx_hop = np.exp(-v * double_hop)
                row_i[j] = jx_hop * np.linalg.det(phi_0[x_hop])
            c_down[x[ind_down+self.getUpNum()]-self.sites] = row_i
      
        correlation_up = c_up / psi_jx
        correlation_down = c_down / psi_jx

        return correlation_up, correlation_down

    def correlation_pair(self,v,phi_0):
        x = self.getX()
        psi_0_x = np.linalg.det(phi_0[self.getX()])
        assert psi_0_x != 0
        c_pair = np.zeros((self.sites, self.sites))
        site_double = self.getDoubleOccupance()
        # print(x, site_double)

        for ind in range(self.getDoubleNum()):
            row_i = np.zeros(self.sites)
            for j in range(self.sites):
                # print(np.argwhere(x==site_double[ind]))
                ind_up = np.argwhere(x==site_double[ind])[0,0]
                ind_down = np.argwhere(x==site_double[ind]+self.sites)[0,0]
                x_hop = x.copy()
                x_hop[ind_up] = j
                x_hop[ind_down] = j+self.sites
                row_i[j] = np.linalg.det(phi_0[x_hop])
            c_pair[self.getDoubleOccupance()[ind]] = row_i

        correlation_pair = c_pair / psi_0_x
        return correlation_pair
    
    def correlation_ns(self,v,phi_0):
        psi_0_x = np.linalg.det(phi_0[self.getX()])
        assert psi_0_x != 0
        
        n_up = np.zeros((self.sites,1))
        n_up[self.up] = 1
        n_down = np.zeros((self.sites,1))
        n_down[self.down] = 1
        n_x = n_up + n_down
        s_x = n_up - n_down
        correlation_n = np.dot(n_x, n_x.T)
        correlation_s = np.dot(s_x, s_x.T)

        return correlation_n, correlation_s

