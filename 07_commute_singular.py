# Twenty commute receipts, every ride 6 km: three tools, three answers.
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
km = np.full(20, 6.0)                         # the same ride every day
price = np.round(9.70 + np.random.default_rng(1).normal(0, 0.3, 20), 2)
X = np.column_stack([np.ones(20), km])
try:
    np.linalg.solve(X.T @ X, X.T @ price)
except np.linalg.LinAlgError as e:
    print("numpy       :", e)
fit = sm.OLS(price, X).fit()
print("statsmodels :", fit.params.round(4))
print(str(fit.summary()).splitlines()[-2:])
lr = LinearRegression().fit(km.reshape(-1, 1), price)
print("scikit-learn:", round(lr.intercept_, 4), lr.coef_.round(4))
