"""
Comparisons.

"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from OA_process import OA_process, performance, read_data
from numpy.linalg import norm
# random.seed(2023)
N = [80 + 5*i for i in range(29)]
results = []
for n in N:
    print(n)
    p = 2000
    k = 10
    rho = 0
    general_w = True
    X, Y, w = read_data(n, p, k, rho, general_w)
    print("Read data completed.")
    gamma = 1 / n
    s0_star, w0_star, s1, time = OA_process(X, Y, k, gamma)
    A, F = performance(w, w0_star)
    results.append([n, A, F, time])
results = pd.DataFrame(results, columns=['n', 'A', 'F', 'Time'])
results.to_csv("results_general_w.csv", index=False)

