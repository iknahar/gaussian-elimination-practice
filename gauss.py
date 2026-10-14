"""Gaussian elimination from scratch, built up one step at a time.

forward_naive  the textbook loop, no row swaps (breaks on tiny pivots)
forward        the grown-up loop: partial pivoting, skips empty columns
back_substitute  solve a staircase from the bottom row up
classify       one answer, no answer, or infinitely many
solve          all of the above in one call
"""
import numpy as np


def forward_naive(M, show=False):
    M = np.array(M, dtype=float)        # a copy, so the receipts stay safe
    n = M.shape[0]
    for c in range(n - 1):              # pivot sits at (c, c)
        for i in range(c + 1, n):       # every row underneath it
            f = M[i, c] / M[c, c]       # the multiplier
            M[i] = M[i] - f * M[c]      # the one row move we need
            if show:
                print(f"R{i+1} <- R{i+1} - {f:g}*R{c+1}")
    return M


def back_substitute(U):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):          # bottom row first
        known = U[i, i + 1:n] @ x[i + 1:]   # the part already solved
        x[i] = (U[i, -1] - known) / U[i, i]
    return x


def forward(M, tol=1e-12):
    M = np.array(M, dtype=float)
    rows, cols = M.shape
    pivots = []                          # (row, column) of every pivot
    r = 0                                # the next row that needs a pivot
    for c in range(cols - 1):            # the last column holds prices
        if r == rows:
            break
        p = r + np.argmax(np.abs(M[r:, c]))  # biggest entry in the column
        if abs(M[p, c]) < tol:           # nothing to pivot on here, so
            continue                     # this unknown will be free
        if p != r:
            M[[r, p]] = M[[p, r]]        # swap it to the top
        for i in range(r + 1, rows):
            f = M[i, c] / M[r, c]
            M[i] = M[i] - f * M[r]
        pivots.append((r, c))
        r += 1
    return M, pivots


def classify(U, pivots, tol=1e-10):
    unknowns = U.shape[1] - 1
    for row in U:
        if np.all(np.abs(row[:-1]) < tol) and abs(row[-1]) > tol:
            return "none"                # this row says 0 = something
    if len(pivots) < unknowns:
        return "infinite"                # a column with no pivot is free
    return "one"


def solve(A, b):
    M = np.column_stack([A, b])
    U, pivots = forward(M)
    kind = classify(U, pivots)
    if kind != "one":
        return kind, None
    n = U.shape[1] - 1
    return kind, back_substitute(U[:n])
