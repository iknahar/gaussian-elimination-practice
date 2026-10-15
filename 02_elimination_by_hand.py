# The three receipts, eliminated one row move at a time, then solved
# from the bottom row up.
import numpy as np
from gauss import back_substitute
M = np.array([[1, 3, 10, 9.10],     # s + 3k + 10m = 9.10
              [1, 5, 14, 12.70],    # s + 5k + 14m = 12.70
              [1, 9, 30, 22.30]])   # s + 9k + 30m = 22.30
M[1] = M[1] - 1 * M[0]              # clear the s under the first pivot
print("R2 <- R2 - 1*R1\n", M.round(2))
M[2] = M[2] - 1 * M[0]
print("R3 <- R3 - 1*R1\n", M.round(2))
M[2] = M[2] - 3 * M[1]              # 6 / 2 = 3 copies of row two
print("R3 <- R3 - 3*R2\n", M.round(2))
print("s, k, m =", back_substitute(M).round(2))
