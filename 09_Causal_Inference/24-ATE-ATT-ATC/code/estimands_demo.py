""" ATE vs ATT vs ATC under effect heterogeneity and selection on benefit."""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(0)
n = 200_000
X = rng.normal(0, 1, (n, 2))
T = rng.binomial(1, 1 / (1 + np.exp(-1.0 * X[:, 0])))           # high x0 -> more likely treated
Y0 = X[:, 0] + X[:, 1] + rng.normal(0, 1, n)
tau = 1 + 2 * X[:, 0]
Y1 = Y0 + tau
Y = np.where(T == 1, Y1, Y0)

true = {"ATE": tau.mean(), "ATT": tau[T == 1].mean(), "ATC": tau[T == 0].mean()}

# G-computation with treatment-covariate interactions (correctly specified)
F = lambda t: np.column_stack([X, t, t[:, None] * X])
m = LinearRegression().fit(F(T.astype(float)), Y)
diff = m.predict(F(np.ones(n))) - m.predict(F(np.zeros(n)))
g = {"ATE": diff.mean(), "ATT": diff[T == 1].mean(), "ATC": diff[T == 0].mean()}

# IPW with estimand-specific weights (Hajek form)
e = LogisticRegression(C=1e6, max_iter=1000).fit(X, T).predict_proba(X)[:, 1]
def hajek(w, mask): return np.sum(w[mask] * Y[mask]) / np.sum(w[mask])
t1, t0 = T == 1, T == 0
ipw = {
    "ATE": hajek(1 / e, t1) - hajek(1 / (1 - e), t0),
    "ATT": Y[t1].mean() - hajek(e / (1 - e), t0),
    "ATC": hajek((1 - e) / e, t1) - Y[t0].mean(),
}

print(f"{'estimand':<8}{'truth':>9}{'g-comp':>9}{'IPW':>9}")
for k in ("ATE", "ATT", "ATC"):
    print(f"{k:<8}{true[k]:>9.3f}{g[k]:>9.3f}{ipw[k]:>9.3f}")
p1 = T.mean()
print(f"\nCheck: P(T=1)*ATT + P(T=0)*ATC = {p1 * true['ATT'] + (1 - p1) * true['ATC']:.3f}  vs ATE = {true['ATE']:.3f}")
