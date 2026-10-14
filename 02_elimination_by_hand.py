# The same three receipts, eliminated one row move at a time.
import numpy as np
from gauss import forward_naive, back_substitute
M = np.array([[1, 3, 10, 9.10],
              [1, 5, 14, 12.70],
              [1, 9, 30, 22.30]])
U = forward_naive(M, show=True)
print(U.round(2))
print("s, k, m =", back_substitute(U).round(2))
