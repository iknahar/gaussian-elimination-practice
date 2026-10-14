# One answer, no answer, infinitely many: the bottom row tells you which.
import numpy as np
from gauss import forward, classify
cases = {
    "two different rides": [[1, 4, 7.30], [1, 10, 14.50]],
    "surge receipt":       [[1, 4, 7.30], [1, 4, 9.80]],
    "same commute twice":  [[1, 6, 9.70], [1, 6, 9.70]],
}
for name, M in cases.items():
    U, pivots = forward(M)
    print(f"{name:22s} last row {U[-1].round(2)}  ->", classify(U, pivots))
