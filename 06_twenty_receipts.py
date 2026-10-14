# Twenty noisy receipts, two unknowns: the normal equations.
import numpy as np
rng = np.random.default_rng(35)
km = np.round(rng.uniform(2, 14, 20), 1)
price = np.round(2.50 + 1.20 * km + rng.normal(0, 0.8, 20), 2)
X = np.column_stack([np.ones(20), km])   # the column of ones again
XtX = X.T @ X                            # 2 x 2
Xty = X.T @ price                        # 2 numbers
print(XtX.round(2), Xty.round(2))
print("normal equations:", np.linalg.solve(XtX, Xty).round(3))
print("lstsq           :", np.linalg.lstsq(X, price, rcond=None)[0].round(3))
