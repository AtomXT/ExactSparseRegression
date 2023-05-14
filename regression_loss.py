"""
This function si Algorithm 2 in the paper.

Compute the regression function c and subgradients.

Inputs:
    Y: n dimension vector
    X: n x p dimension data matrix
    s: in binary set S_k^p which contains p dimensional binary vector with at most k nonzero elements.
    gamma: Tikhonov regularization term
Output:
    c: regression function objective value
    d_c: gradient of regression function

"""
import numpy as np


def regression_loss(X, Y, s, gamma):
    """

    :param X: n by p numpy array
    :param Y: n by 1 numpy array
    :param s: p by 1 binary vector
    :param gamma: Tikhonov regularization term
    :return:
        c: regression function objective value
        d_c: gradient of regression function
    """

    # m = gp.Model()
    # n, p = X.shape
    # alpha = m.addMVar((n, 1), lb=-GRB.INFINITY, vtype='c')
    # obj1 = 0
    # for j in range(p):
    #     Kj = np.outer(X[:, j], X[:, j])
    #     obj1 += s[j] * (alpha @ Kj @ alpha) * -gamma/2
    # obj2 = -alpha @ alpha/2
    # obj3 = Y.T @ alpha
    # obj = obj1 + obj2 + obj3
    #
    # m._alpha = alpha
    # m.setObjective(obj, GRB.MAXIMIZE)
    # m.optimize()
    n, p = X.shape
    k = sum(s)
    Xs = X[:, np.array(s) == 1]
    Ik = np.eye(k)
    alpha = Y - Xs @ np.linalg.inv(Ik/gamma + Xs.T @ Xs) @ Xs.T @ Y
    c = Y.T @ alpha / 2
    d_c = np.zeros((p, 1))
    for j in range(p):
        d_c[j] = -gamma/2 * (X[:, j]@alpha)**2
    return c, d_c


if __name__ == '__main__':
    n = 50
    p = 5
    X = np.random.randn(n, p)
    w = np.random.randint(2, size=5)
    Y = X @ w
    gamma = 1e6
    c, d_c = regression_loss(X, Y, [1] * p, gamma)
    print(c, d_c)
    print(w@w/2/gamma)



