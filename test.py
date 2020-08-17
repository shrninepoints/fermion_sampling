import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn
from scipy.linalg import qr
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
import time
import fermion_sampling as fs

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args[0:2], kw, te-ts))
        return result
    return timed

@timeit
def builtin():
    result = np.zeros(L)
    for _ in range(1):
        DPP.sample_exact()
        m = [1 if i in DPP.list_of_samples[-1] else 0 for i in range(L)]
        result = np.add(result, m)
    print(result / 1)

N, L = 512, 4096

# Random orthogonal vectors
eig_vecs, _ = qr(randn(L, N), mode='economic')
for v in eig_vecs:
    v = v / np.dot(v,v)
# Random eigenvalues
#eig_vals = rand(N)  # 0< <1
eig_vals = np.ones(N) # 0 or 1 i.e. projection
DPP = FiniteDPP('correlation',
                **{'K_eig_dec': (eig_vals, eig_vecs)})# Sample
#builtin()
U = eig_vecs.T
fs.sample_position(1,'sampling',U,N,L)

pass
