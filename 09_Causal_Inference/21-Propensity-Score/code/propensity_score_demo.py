""" Propensity score: balance diagnostics and stratification."""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 60_000
X = rng.normal(0, 1, (n, 3))
T = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2]))))
Y = 2 * T + 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2] + rng.normal(0, 1, n)   # true ATE = 2

e = LogisticRegression(C=1e6, max_iter=1000).fit(X, T).predict_proba(X)[:, 1]

def smd(x, t, w=None):
    w = np.ones_like(x) if w is None else w
    m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1]); v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)

w = np.where(T == 1, 1 / e, 1 / (1 - e))
strata = np.digitize(e, np.quantile(e, [0.2, 0.4, 0.6, 0.8]))

def strat_smd(j):
    tot = 0.0
    for s in range(5):
        m = strata == s
        tot += m.mean() * smd(X[m, j], T[m])
    return tot

print(f"{'covariate':<11}{'raw SMD':>10}{'IPW SMD':>10}{'5-strata SMD':>14}")
for j in range(3):
    print(f"X{j:<10}{smd(X[:, j], T):>10.3f}{smd(X[:, j], T, w):>10.3f}{strat_smd(j):>14.3f}")

est = sum((strata == s).mean() * (Y[(strata == s) & (T == 1)].mean() - Y[(strata == s) & (T == 0)].mean()) for s in range(5))
print(f"\nNaive ATE                 = {Y[T==1].mean() - Y[T==0].mean():.3f}")
print(f"PS-stratified ATE (5)     = {est:.3f}   (true 2.000)")
print(f"Propensity range: treated [{e[T==1].min():.3f}, {e[T==1].max():.3f}], control [{e[T==0].min():.3f}, {e[T==0].max():.3f}]")
