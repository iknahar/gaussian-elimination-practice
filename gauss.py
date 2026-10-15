"""Gaussian elimination from scratch, built up one step at a time.

eliminate        turn the grid into a staircase (pivot=False is the
                 plain textbook version that breaks on tiny pivots)
back_substitute  solve a staircase from the bottom row up
classify         one answer, no answer, or infinitely many
solve            all of the above in one call
"""
import numpy as np


def eliminate(M, pivot=True, tol=1e-12):
    M = np.array(M, dtype=float)          # a copy, the receipts stay safe
    rows, cols = M.shape
    pivots, r = [], 0                     # pivot columns, next free row
    for c in range(cols - 1):             # the last column holds prices
        if r == rows:
            break
        if np.max(np.abs(M[r:, c])) < tol:  # nothing usable here, so
            continue                        # this unknown is free
        p = r                             # textbook: take the next row
        if pivot:                         # better: take the biggest number
            p = r + np.argmax(np.abs(M[r:, c]))
        M[[r, p]] = M[[p, r]]             # move that row to the top
        for i in range(r + 1, rows):
            f = M[i, c] / M[r, c]         # the multiplier
            M[i] = M[i] - f * M[r]        # the one row move we need
        pivots.append(c)
        r += 1
    return M, pivots


def back_substitute(U):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):          # bottom row first
        known = U[i, i + 1:n] @ x[i + 1:]   # the part already solved
        x[i] = (U[i, -1] - known) / U[i, i]
    return x


def classify(U, pivots, tol=1e-10):
    unknowns = U.shape[1] - 1
    for row in U:
        if np.all(np.abs(row[:-1]) < tol) and abs(row[-1]) > tol:
            return "none"                # this row says 0 = something
    if len(pivots) < unknowns:
        return "infinite"                # a column with no pivot is free
    return "one"


def solve(A, b):
    U, pivots = eliminate(np.column_stack([A, b]))
    kind = classify(U, pivots)
    if kind != "one":
        return kind, None
    n = U.shape[1] - 1
    return kind, back_substitute(U[:n])
