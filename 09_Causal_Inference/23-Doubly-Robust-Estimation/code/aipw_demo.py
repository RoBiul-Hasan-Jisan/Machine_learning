""" AIPW: outcome model and propensity model each right or wrong."""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(0)
n = 100_000
X = rng.normal(0, 1, (n, 2))
logit_e = 0.8 * X[:, 0] + 1.0 * (X[:, 1] ** 2 - 1)
T = rng.binomial(1, 1 / (1 + np.exp(-logit_e)))
Y = 2 * T + X[:, 0] + 1.5 * X[:, 1] ** 2 + rng.normal(0, 1, n)        # true ATE = 2

def feats(X, right):
    return np.column_stack([X, X[:, 1] ** 2]) if right else X

def estimators(out_right, ps_right):
    F = feats(X, out_right)
    m1 = LinearRegression().fit(F[T == 1], Y[T == 1]).predict(F)
    m0 = LinearRegression().fit(F[T == 0], Y[T == 0]).predict(F)
    G = feats(X, ps_right)
    e = LogisticRegression(C=1e6, max_iter=2000).fit(G, T).predict_proba(G)[:, 1]
    gcomp = np.mean(m1 - m0)
    ipw = np.mean(T * Y / e) - np.mean((1 - T) * Y / (1 - e))
    aipw = np.mean(m1 + T * (Y - m1) / e) - np.mean(m0 + (1 - T) * (Y - m0) / (1 - e))
    return gcomp, ipw, aipw

print("True ATE = 2.000\n")
print(f"{'outcome model':<16}{'propensity':<14}{'g-comp':>9}{'IPW':>9}{'AIPW':>9}")
for o in (True, False):
    for p in (True, False):
        g, i, a = estimators(o, p)
        print(f"{'correct' if o else 'WRONG':<16}{'correct' if p else 'WRONG':<14}{g:>9.3f}{i:>9.3f}{a:>9.3f}")
