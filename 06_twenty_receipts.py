# Noisy receipts, two unknowns: least squares and the normal equations.
import numpy as np
rng = np.random.default_rng(35)
km = np.round(rng.uniform(2, 14, 20), 1)
price = np.round(2.50 + 1.20 * km + rng.normal(0, 0.8, 20), 2)
X = np.column_stack([np.ones(20), km])   # the column of ones again
print("normal equations:", np.linalg.solve(X.T @ X, X.T @ price).round(3))
print("lstsq           :", np.linalg.lstsq(X, price, rcond=None)[0].round(3))
# more receipts, better guess (a fresh pile of 200)
rng = np.random.default_rng(7)
km = np.round(rng.uniform(2, 14, 200), 1)
price = np.round(2.50 + 1.20 * km + rng.normal(0, 0.8, 200), 2)
for n in [5, 20, 50, 200]:
    X = np.column_stack([np.ones(n), km[:n]])
    s, k = np.linalg.lstsq(X, price[:n], rcond=None)[0]
    print(f"{n:3d} receipts: per km guess {k:.3f}")
