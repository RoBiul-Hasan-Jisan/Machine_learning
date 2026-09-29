""" G-computation with a misspecified vs a flexible outcome model."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)

def simulate(n):
    X = rng.normal(0, 1, (n, 2))
    # treatment also depends on X0^2, so the omitted quadratic term is correlated with T
    p = 1 / (1 + np.exp(-(0.8 * X[:, 0] + 0.5 * X[:, 1] + 1.0 * (X[:, 0] ** 2 - 1))))
    T = rng.binomial(1, p)
    Y0 = X[:, 0] + 0.5 * X[:, 0] ** 2 + X[:, 1] + rng.normal(0, 1, n)
    Y1 = Y0 + 1 + 0.5 * X[:, 0]
    return X, T, np.where(T == 1, Y1, Y0)

def gcomp(X, T, Y, flexible):
    def feats(X, T):
        base = np.column_stack([X, T])
        if not flexible:
            return base
        poly = PolynomialFeatures(2, include_bias=False).fit_transform(X)
        return np.column_stack([poly, T, T[:, None] * poly])
    m = LinearRegression().fit(feats(X, T), Y)
    n = len(Y)
    return m.predict(feats(X, np.ones(n))).mean() - m.predict(feats(X, np.zeros(n))).mean()

X, T, Y = simulate(50_000)
print(f"True ATE                          = 1.000")
print(f"Naive difference in means         = {Y[T==1].mean() - Y[T==0].mean():.3f}")
print(f"G-computation, linear (misspec.)  = {gcomp(X, T, Y, False):.3f}")
print(f"G-computation, quadratic+interact = {gcomp(X, T, Y, True):.3f}")

Xs, Ts, Ys = simulate(3_000)
boots = []
for _ in range(300):
    idx = rng.integers(0, len(Ys), len(Ys))
    boots.append(gcomp(Xs[idx], Ts[idx], Ys[idx], True))
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\nn=3,000 flexible g-comp: {gcomp(Xs, Ts, Ys, True):.3f}, bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
