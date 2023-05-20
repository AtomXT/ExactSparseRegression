"""
Generate data as in section 4.1.

- Y = X @ w_true + E
- Nonzero coefficients in w_true are drwan uniformly at random from the set {-1 ,1}.
- S := X @ w_true
- E ~ N(0, sigma^2) sqrt(SNR) = norm(S)/norm(E). Set sqrt(SNR) = 20.
- X ~ N(0, Sigma) Sigma(i,j) = rho^(|i-j|), rho = 0.1
"""
import random
import numpy as np
from numpy.linalg import norm
# random.seed(2023)


general_w = True  # generate w whose nonzero values are uniformly distributed in [0.8, 1.2] if True.

# N = [80 + 5*i for i in range(29)]
N = [95]
for n in N:
    print(n)
    # n = 100  # number of samples
    p = 2000  # data dimension
    k = 10  # number of nonzero coefficients in w_true, let n/k = 1000 as in Table 1
    rho = 0  # correlation coefficient
    Sigma = np.array([[rho**(abs(i-j)) for j in range(p)] for i in range(p)])  # variance matrix of X
    X = np.random.multivariate_normal(np.zeros(p), Sigma, n)  # n by p matrix
    w = np.zeros(p)
    nonzero_ind = np.random.choice(range(p), k, replace=False)  # choose k nonzero elements
    negative_ind = np.random.choice(nonzero_ind, k//2)  # k//2 of them are -1
    if general_w:
        w[nonzero_ind] = np.random.uniform(0.5, 1.5, k)
    else:
        w[nonzero_ind] = 1
    w[negative_ind] = -w[negative_ind]
    S = X @ w
    error = np.random.randn(n)
    Y = X @ w + error/20*norm(S)/norm(error)

    if general_w:
        data_path = "./data/general_"
    else:
        data_path = "./data/"
    np.savetxt(data_path + "data_%s_%s_%s_%s.txt" % (n, p, k, rho), X)
    np.savetxt(data_path + "Y_%s_%s_%s_%s.txt" % (n, p, k, rho), Y)
    np.savetxt(data_path + "w_true_%s_%s_%s_%s.txt" % (n, p, k, rho), w)

