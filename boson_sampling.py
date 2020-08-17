import numpy as np

x = np.array([[0.5, -0.8660254037844386], [0.8660254037844386, 0.5]], dtype=complex)  # input matrix
res = [[0, 0], [0, 1], [1, 0], [1, 1]]   # input all possible results


def subperm(x):         # 计算所有余子式
    n = np.size(x, 0)   # dimesion
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(0, n, 1)
    v = np.sum(x, 0)  # 遇到一列和为零怎么办？
    ps = np.prod(v)

    if ps == 0:
        ind = np.where(v == 0)
        ind = ind[0]
        p = np.zeros(n+1)
        if len(ind) == 1:
            v1 = v.copy()
            v1[ind] = 1
            p[ind] = np.prod(v1)
    else:
        p = np.array([ps/v[i] for i in range(n+1)])

    while j < (n-1):
        v = v - 2 * d[j] * x[j]
        d[j] = -d[j]
        s = -s
        prs = np.prod(v)
        if prs == 0:
            ind = np.where(v == 0)
            ind = ind[0]
            pr = np.zeros(n + 1)
            if len(ind) == 1:
                v1 = v.copy()
                v1[ind] = 1
                pr[ind] = np.prod(v1)
        else:
            pr = np.array([prs / v[i] for i in range(n + 1)])

        p = p + s * pr
        f[0] = 0         # Gray code order
        f[j] = f[j+1]
        f[j+1] = j+1
        j = f[0]
    y = p / (2 ** (n-1))
    return y


def sample(x):
    N = np.size(x, 1)
    M = np.size(x, 0)
    a = np.random.permutation(N)
    r = x[:, a[0]]
    prob = r * np.conjugate(r)
    prob = prob.real
    r0 = np.random.choice(M, 1, p=prob)
    row = r0
    for i in range(1, N):
        column = a[0:i+1]
        submat = x[np.ix_(row, column)]
        sub = subperm(submat)
        prob = []
        div = np.prod(range(1, i + 2)) # factorial of k
        for j in range(M):
            perm = np.dot(x[[j], column], sub)
            p = np.square(np.abs(perm)) / div
            prob.append(p.real)
        probs = sum(prob)
        prob = prob / probs
        ri = np.random.choice(M, 1, p=prob)
        row = np.stack((row, ri), 1)
        print(row)
    return row[0]


result = []
for i in range(100):
    result.append(list(sample(x)))

count = [result.count(i)/100 for i in res]

print(res)
print(count)
