# 0.1 + 0.2, then a pivot of 1e-20: fine by hand, wrong on a computer,
# fixed by a row swap.
import numpy as np
from gauss import eliminate, back_substitute
print(0.1 + 0.2)                     # 0.30000000000000004
M = np.array([[1e-20, 1.0, 1.0],     # 0.00000000000000000001x + y = 1
              [1.0, 1.0, 2.0]])      #                       x + y = 2
U, _ = eliminate(M, pivot=False)     # the plain textbook way
print(back_substitute(U))            # [0. 1.]
U, _ = eliminate(M)                  # biggest number goes on top first
print(back_substitute(U))            # [1. 1.]
print(1 - 1e20 == -1e20)             # True, the 1 fell off the end
