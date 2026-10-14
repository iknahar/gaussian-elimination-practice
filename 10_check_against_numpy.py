# Our solver against numpy on 1,000 random systems, plus the edge cases.
import numpy as np
from gauss import solve
rng = np.random.default_rng(0)
worst = 0.0
for trial in range(1000):
    n = int(rng.integers(2, 30))
    A = rng.normal(size=(n, n))
    b = rng.normal(size=n)
    kind, x = solve(A, b)
    worst = max(worst, np.max(np.abs(x - np.linalg.solve(A, b))))
print(f"worst gap from numpy over 1,000 systems: {worst:.1e}")
print(solve([[1, 4], [1, 4]], [7.30, 9.80]))   # surge
print(solve([[1, 6], [1, 6]], [9.70, 9.70]))   # commute
print(solve([[1e-20, 1], [1, 1]], [1, 2]))      # tiny pivot
