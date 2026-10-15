# The same algorithm, written as a Python loop, against numpy's version.
import time
import numpy as np
from gauss import solve
rng = np.random.default_rng(0)
for n in [50, 200, 800]:
    A = rng.standard_normal((n, n)); b = rng.standard_normal(n)
    t = time.perf_counter(); solve(A, b); ours = time.perf_counter() - t
    t = time.perf_counter(); np.linalg.solve(A, b); theirs = time.perf_counter() - t
    print(f"n={n:4d} our loop {ours*1000:8.2f} ms  numpy {theirs*1000:7.3f} ms")
