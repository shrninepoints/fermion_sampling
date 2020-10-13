import numpy as np
L = 2 ** 10
u = np.random.rand(L)
u3 = np.random.rand(L)
diff1 = np.dot(u,u) / np.linalg.norm(u)**2
u2 = ((np.random.rand(L) - 0.5) / 10 ** 0 + 1) * u
diff2 = np.dot(u,u2) / (np.linalg.norm(u)*np.linalg.norm(u2))

print(u)
print(np.dot(u,u3)/(np.linalg.norm(u)*np.linalg.norm(u3)))
print(diff1,diff2)
print(np.log(diff1-diff2))