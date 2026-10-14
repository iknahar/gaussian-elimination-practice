# Gaussian elimination practice

Practice code for learning systems of linear equations and Gaussian
elimination. Everything starts from three ride receipts and a fare formula:

```
price = start fee + (price per km) x km + (price per minute) x minutes
```

Three receipts give three equations, and solving them recovers the three
prices. From there the scripts build an elimination solver from scratch,
break it on purpose, fix it, and check it against numpy.

## Setup

```bash
python -m pip install -r requirements.txt
```

Every script runs on its own from this folder, for example:

```bash
python 02_elimination_by_hand.py
```

## What is in here

| file | what it shows |
|---|---|
| `gauss.py` | the solver: `forward_naive`, `forward` (with pivoting), `back_substitute`, `classify`, `solve` |
| `01_receipts_one_liner.py` | three receipts solved with `np.linalg.solve` |
| `02_elimination_by_hand.py` | the same receipts, every row move printed |
| `03_three_outcomes.py` | one answer, no answer (surge), infinitely many (commute) |
| `04_sympy_rref.py` | exact reduced row echelon form and a general solution |
| `05_tiny_pivot.py` | a pivot of `1e-20` gives `x = 0` without a row swap |
| `06_twenty_receipts.py` | twenty noisy receipts, the normal equations, `lstsq` |
| `07_commute_singular.py` | same distance every ride: numpy, statsmodels and scikit-learn disagree |
| `08_solve_vs_inverse.py` | solve against invert-then-multiply, on time and accuracy |
| `09_lu_reuse.py` | factor once with `lu_factor`, then solve 500 price lists |
| `10_check_against_numpy.py` | our solver against numpy on 1,000 random systems |

Timings depend on your machine and on what else it is doing. The ratios are
the part worth comparing.

## Exercises

1. Change the third receipt in `01_receipts_one_liner.py` from $22.30 to
   $25.00. Run it again. Do the new prices still look like a fare?

<details><summary>Answer</summary>

The solver still finds one answer, because the three rides are still
different: start fee 1.15, per km 0.525, per minute 0.6375. The system has
no idea the prices look odd. One overpriced receipt (a surge, a tip) drags
all three numbers, which is why real fare decoding uses many receipts and
least squares instead of exactly three.
</details>

2. Add a fourth ride of 2 km and 8 minutes. Classify the system when it cost
   $7.30, then when it cost $8.00. Use `forward` and `classify`.

<details><summary>Answer</summary>

At $7.30 the fourth receipt agrees with the other three (2.50 + 2.40 + 2.40),
so elimination ends in a row of zeros and the answer is still `one`. At
$8.00 it disagrees, the last row becomes `0 = 0.70`, and the answer is
`none`. Four equations and three unknowns can go either way.
</details>

3. Fix the commute problem from `07_commute_singular.py` with just two
   receipts: one ride of 6 km for $9.70 and one of 8 km for $12.10.

<details><summary>Answer</summary>

`solve([[1, 6], [1, 8]], [9.70, 12.10])` returns start fee 2.50 and 1.20
per km. Two different distances are enough to separate the start fee from
the per-km price. The cure for "Singular matrix" here is variety in the
data, not a cleverer solver.
</details>

4. Hilbert matrices are famously hard. Build one with
   `scipy.linalg.hilbert(n)` for n = 8 and n = 12, set `b = H @ ones`, and
   compare `np.linalg.solve` with `np.linalg.inv(H) @ b`. Look at both the
   relative miss `|Hx - b| / |b|` and the error in x.

<details><summary>Answer</summary>

For n = 12, solve misses b by about 1e-16 but its x is off by about 0.7.
The inverse misses b by about 2e-2 and its x is off by about 21. Solving
still wins, but a tiny miss does not promise a correct x when the matrix
is this close to singular. That gap is what the condition number measures.
</details>

5. Run `scipy.linalg.lu` on the receipts matrix
   `[[1, 3, 10], [1, 5, 14], [1, 9, 30]]`. Why does its U differ from the
   staircase in `02_elimination_by_hand.py`?

<details><summary>Answer</summary>

SciPy uses partial pivoting. After the first column is cleared, the second
column holds 2 and 6, so it swaps rows to pivot on the bigger 6. That is
why P is not the identity and U is `[[1, 3, 10], [0, 6, 20], [0, 0, -2.667]]`.
Different staircase, same answer.
</details>

## License

MIT
