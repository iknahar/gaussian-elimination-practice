# Factor once, solve many: one ride matrix, 500 different price lists.
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
rng = np.random.default_rng(0)
A = rng.standard_normal((1000, 1000))
B = rng.standard_normal((1000, 500))     # 500 price lists
t = time.perf_counter()
for j in range(500):
    np.linalg.solve(A, B[:, j])          # eliminates from scratch each time
print(f"solve each time : {time.perf_counter() - t:.3f} s")
t = time.perf_counter()
lu, piv = lu_factor(A)                   # the slow part, once
for j in range(500):
    lu_solve((lu, piv), B[:, j])         # two quick triangle sweeps
print(f"factor once     : {time.perf_counter() - t:.3f} s")
