"""
Solve the continuous relaxation of the sparse kernel regression problem (10) can be reduced to the following SOCP:

Use the optimal solution alpha and its relation to s to pick a warm start s vector.

"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def warm_start(X, Y, k, gamma):
    m = gp.Model()
    n, p = X.shape


    alpha = {}
    for i in range(n):
        alpha[i] = m.addVar(lb=-GRB.INFINITY, vtype='c', name='alpha%s' % i)

    t = m.addVar(lb=-GRB.INFINITY, name='t')
    u = {}
    for i in range(p):
        u[i] = m.addVar(vtype='c', name='u%s' % i)

    for j in range(p):
        temp = gp.quicksum(X[i, j]*alpha[i] for i in range(n))
        m.addConstr(2*u[j]/gamma >= temp * temp - 2*t/gamma)

    obj = -1/2 * gp.quicksum(alpha[j]*alpha[j] for j in range(n)) + gp.quicksum(Y[j]*alpha[j] for j in range(n)) - \
          gp.quicksum(u[i] for i in range(p)) - k*t

    m._alpha = alpha
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    alpha_val = np.array([var.X for var in m.getVars() if "alpha" in var.VarName])
    aka = [(alpha_val @ X[:, j])**2 for j in range(p)]
    ind = np.argpartition(aka, -k)[-k:]
    s = np.zeros((p, 1))
    s[ind] = 1
    return s


if __name__ == '__main__':
    n = 50
    p = 5
    X = np.random.randn(n, p)
    w = np.random.randint(2, size=5)
    Y = X @ w
    gamma = 1e6
    alpha_val = warm_start(X, Y, 2, gamma)
    print(alpha_val)
    print(w)