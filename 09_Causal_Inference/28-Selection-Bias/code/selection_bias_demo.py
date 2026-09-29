""" Selection bias: collider selection vs selection on an effect modifier (IPSW)."""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 600_000
sig = lambda z: 1 / (1 + np.exp(-z))

print("=== (A) Randomized T, but selection S depends on T and Y (collider) ===")
T = rng.binomial(1, 0.5, n)
Y = 2.0 * T + rng.normal(0, 1, n)                          # true effect 2, no confounding
S = rng.binomial(1, sig(-1 + 1.5 * T + 1.0 * Y)).astype(bool)
print(f"population diff-in-means   = {Y[T == 1].mean() - Y[T == 0].mean():.3f}   (truth 2.000)")
print(f"selected-sample diff       = {Y[S & (T == 1)].mean() - Y[S & (T == 0)].mean():.3f}   <- biased by conditioning on S")

print("\n=== (B) Randomized T, S depends on an effect modifier X (tau = 1 + X, population ATE = 1) ===")
X = rng.normal(0, 1, n)
T = rng.binomial(1, 0.5, n)
Y = (1 + X) * T + X + rng.normal(0, 1, n)
S = rng.binomial(1, sig(-0.5 + 1.5 * X)).astype(bool)
naive = Y[S & (T == 1)].mean() - Y[S & (T == 0)].mean()
p_sel = LogisticRegression(C=1e6, max_iter=1000).fit(X[:, None], S).predict_proba(X[:, None])[:, 1]
w = 1 / p_sel
def wmean(mask): return np.sum(w[mask] * Y[mask]) / np.sum(w[mask])
ipsw = wmean(S & (T == 1)) - wmean(S & (T == 0))
print(f"selected-sample diff       = {naive:.3f}   <- effect in the selected (high-X) group")
print(f"IPSW estimate              = {ipsw:.3f}   (population ATE = 1.000)")
