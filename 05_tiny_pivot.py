# A pivot of 1e-20: perfect by hand, wrong on a computer, fixed by a swap.
import numpy as np
from gauss import forward_naive, forward, back_substitute
eps = 1e-20
M = np.array([[eps, 1.0, 1.0],      # eps*x + y = 1
              [1.0, 1.0, 2.0]])     #     x + y = 2
print("no swap  :", back_substitute(forward_naive(M)))
print("with swap:", back_substitute(forward(M)[0]))
print("numpy    :", np.linalg.solve(M[:, :2], M[:, 2]))
print("1 - 1e20 == -1e20 ?", 1 - 1e20 == -1e20)
