# Exact fractions with sympy: the reduced form and the general solution.
import sympy as sp
R = sp.Rational
M = sp.Matrix([[1, 3, 10, R("9.10")],
               [1, 5, 14, R("12.70")],
               [1, 9, 30, R("22.30")]])
rref, pivots = M.rref()
print(rref, pivots)
# the commute: two copies of one receipt, so one unknown is free
s, k = sp.symbols("s k")
print(sp.linsolve([s + 6*k - R("9.70"), s + 6*k - R("9.70")], [s, k]))
