# ExactSparseRegression
Python implementation for the cutting plane algorithm in: [Sparse high-dimensional regression: Exact scalable algorithms and phase transitions](https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-1/Sparse-high-dimensional-regression--Exact-scalable-algorithms-and-phase/10.1214/18-AOS1804.short)

Step1: 
Run data_generation.py with specific parameters: 
  - n: number of samples
  - p: dimension
  - k: number of nonzero components in the true regressor $w$
  - $\rho$: correlation coefficient

Step2:
Run OA_process.py to get the results.

Note that: for large problem, turn off the warm-start attribute. i.e. s0_star, w0_star, s1 = OA_process(X, Y, k, gamma, False)
