# Solve versus invert-then-multiply: time, and how many digits survive.
import time
import numpy as np
rng = np.random.default_rng(0)
def best(f, k=5):
    times = []
    for _ in range(k):
        t = time.perf_counter(); f(); times.append(time.perf_counter() - t)
    return min(times) * 1000                  # milliseconds
for n in [100, 500, 1000, 2000]:
    A = rng.standard_normal((n, n)); b = rng.standard_normal(n)
    ts = best(lambda: np.linalg.solve(A, b))
    ti = best(lambda: np.linalg.inv(A) @ b)
    print(f"n={n:5d} solve {ts:8.2f} ms  inverse {ti:8.2f} ms  x{ti/ts:.1f}")
# a nasty matrix: condition number 1e14
n = 200
U, _ = np.linalg.qr(rng.standard_normal((n, n)))
V, _ = np.linalg.qr(rng.standard_normal((n, n)))
A = U @ np.diag(np.logspace(0, -14, n)) @ V.T
b = A @ rng.standard_normal(n)
for name, x in [("solve", np.linalg.solve(A, b)),
                ("inverse", np.linalg.inv(A) @ b)]:
    miss = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    print(f"{name:8s} relative miss {miss:.1e}")
