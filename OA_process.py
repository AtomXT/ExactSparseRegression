"""
This is Algorithm 1 in the paper.
Input:
    Y: n dimension vector
    X: n x p dimension data matrix
    k: integer from 1 to p. It is the number of nonzero coefficients in w
Output:
    s_0^* in binary set S_k^p which contains p dimensional binary vector with at most k nonzero elements.
    w_0^*: p dimensional vector. Our estimated coefficient vector.

"""


import gurobipy as gp
import numpy as np
from gurobipy import GRB
import timeit
import random

from regression_loss import regression_loss
from warm_start import warm_start


def performance(true_w, estimated_w):
    """
    Compute the true positive rate and false positive rate as in the paper.
    :param true_w: True w
    :param estimated_w: Estimated w
    :return:
        tpr: true positive rate
        fpr: false positive rate
    """
    true_w, estimated_w = np.sign(true_w), np.sign(estimated_w)
    k, p = np.count_nonzero(true_w), len(true_w)
    tpr = sum((np.multiply(np.sign(estimated_w).T, true_w) == 1)[0])/k
    fpr = sum(np.logical_and(np.sign(estimated_w).T != 0, true_w == 0)[0])/(p-k)
    return tpr, fpr


def read_data(n, p, k, rho, general=False):
    if general:
        data_path = "./data/general_"
        dtype = float
    else:
        data_path = "./data/"
        dtype = int
    data_name = "data_%s_%s_%s_%s.txt" % (n, p, k, rho)
    Y_name = "Y_%s_%s_%s_%s.txt" % (n, p, k, rho)
    w_name = "w_true_%s_%s_%s_%s.txt" % (n, p, k, rho)
    X = np.loadtxt(data_path + data_name)
    Y = np.loadtxt(data_path + Y_name)
    w = np.loadtxt(data_path + w_name, dtype=dtype)
    return X, Y, w


def OA_process(X, Y, k, gamma, ws=False):
    s1 = None
    if ws:
        s1 = warm_start(X, Y, k, gamma)
        print("Warm start ready.")

    m = gp.Model()
    n, p = X.shape
    s = {}
    for j in range(p):
        s[j] = m.addVar(vtype=GRB.BINARY, name='s%s' % j)
    eta = m.addVar(lb=1e-3, vtype='c', name='eta')

    m.addConstr(gp.quicksum(s[j] for j in range(p)) == k)

    # define the callback function
    def logarithmic_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # Get the value of s
            s_val = model.cbGetSolution(model._s)
            s_val = [int(i) for i in s_val.values()]
            c, d_c = regression_loss(X, Y, s_val, gamma)
            rhs = c
            for j in range(p):
                rhs += d_c[j][0]*(model._s[j]-s_val[j])
            model.cbLazy(model._eta >= rhs)

    m._s = s
    m._eta = eta
    m.setObjective(eta, GRB.MINIMIZE)
    if ws:
        m.NumStart = 1
        m.params.StartNumber = 0

        # now set MIP start values using the Start attribute, e.g.:
        for i in range(p):
            s[i].Start = s1[i]
    m.Params.lazyConstraints = 1
    m.Params.TIME_LIMIT = 300
    start = timeit.default_timer()
    m.optimize(logarithmic_callback)
    end = timeit.default_timer()

    s0_star = np.array([round(var.X) for var in m.getVars() if "s" in var.VarName])
    w0_star = np.zeros((p, 1))
    X_s0 = X[:, s0_star == 1]
    w0_star[s0_star == 1, 0] = np.linalg.inv(np.eye(k)/gamma + X_s0.T @ X_s0) @ X_s0.T @ Y
    return s0_star, w0_star, s1, end-start


if __name__ == '__main__':
    # read data
    n = 95
    p = 2000
    k = 10
    rho = 0
    general_w = True
    X, Y, w = read_data(n, p, k, rho, general_w)
    # print("Read data completed.")
    gamma = 1/n
    s0_star, w0_star, s1, time = OA_process(X, Y, k, gamma)
    A, F = performance(w, w0_star)
    print(A, F)
    print(w0_star[w != 0].T)
    print(w[w != 0])
    print(time)
    # print(s1)
