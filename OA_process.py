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
    m.optimize(logarithmic_callback)

    s0_star = np.array([round(var.X) for var in m.getVars() if "s" in var.VarName])
    w0_star = np.zeros((p, 1))
    X_s0 = X[:, s0_star == 1]
    w0_star[s0_star == 1, 0] = np.linalg.inv(np.eye(k)/gamma + X_s0.T @ X_s0) @ X_s0.T @ Y
    return s0_star, w0_star, s1


if __name__ == '__main__':
    # read data
    n = 1000
    p = 500
    k = 5
    rho = 0
    data_path = "./data/"
    data_name = "data_%s_%s_%s_%s.txt" % (n, p, k, rho)
    Y_name = "Y_%s_%s_%s_%s.txt" % (n, p, k, rho)
    w_name = "w_true_%s_%s_%s_%s.txt" % (n, p, k, rho)
    X = np.loadtxt(data_path + data_name)
    Y = np.loadtxt(data_path + Y_name)
    w = np.loadtxt(data_path + w_name, dtype=int)
    print("Read data completed.")
    gamma = 1/n
    s0_star, w0_star, s1 = OA_process(X, Y, k, gamma, True)
    print((np.sign(w0_star).T@np.sign(w))/k)
    print(w0_star[w!=0].T)
    print(w[w!=0])
    # print(s1)
