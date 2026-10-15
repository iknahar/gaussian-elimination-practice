# Where is the phone? Three towers, three distances. Each distance is a
# circle; subtracting one circle's equation from another cancels the
# squared terms and leaves a straight line, so elimination can finish it.
import numpy as np
towers = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]])   # km
phone = np.array([1.5, 1.0])                              # the secret
r = np.linalg.norm(towers - phone, axis=1)                # what we measure
# circle i: (x - xi)^2 + (y - yi)^2 = ri^2. Subtract circle 0 from 1 and 2.
A = 2 * (towers[1:] - towers[0])
b = r[0]**2 - r[1:]**2 - towers[0] @ towers[0] + np.sum(towers[1:]**2, axis=1)
print("distances:", r.round(3))
print("phone found at:", np.linalg.solve(A, b).round(3))
