# Three ride receipts, three unknown prices, one line of numpy.
import numpy as np
# each row is one ride: [start fee, km, minutes]
A = np.array([[1, 3, 10],
              [1, 5, 14],
              [1, 9, 30]], dtype=float)
b = np.array([9.10, 12.70, 22.30])     # what I actually paid
x = np.linalg.solve(A, b)
print("start fee, per km, per minute:", x.round(2))
