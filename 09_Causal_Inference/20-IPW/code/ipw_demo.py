""" IPW: Horvitz-Thompson vs Hajek, weight diagnostics, truncation."""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

def run(strength, n=20_000, trunc=None):
    X = rng.normal(0, 1, (n, 2))
    p = 1 / (1 + np.exp(-strength * (X[:, 0] + 0.5 * X[:, 1])))
    T = rng.binomial(1, p)
    Y = 2.0 * T + 2 * X[:, 0] + X[:, 1] + rng.normal(0, 1, n)        # true ATE = 2
    e = LogisticRegression(C=1e6, max_iter=1000).fit(X, T).predict_proba(X)[:, 1]
    if trunc:                      # clip propensity scores to [trunc, 1 - trunc]
        e = np.clip(e, trunc, 1 - trunc)
    w = np.where(T == 1, 1 / e, 1 / (1 - e))
    ht = np.mean(T * Y / e) - np.mean((1 - T) * Y / (1 - e))
    haj = (np.sum(T * Y / e) / np.sum(T / e)) - (np.sum((1 - T) * Y / (1 - e)) / np.sum((1 - T) / (1 - e)))
    ess = w.sum() ** 2 / (w ** 2).sum()
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    return naive, ht, haj, ess, w.max()

print("True ATE = 2.0\n")
print(f"{'X->T strength':<16}{'naive':>8}{'HT':>8}{'Hajek':>8}{'ESS':>10}{'max w':>9}")
for s in (0.5, 1.5, 3.0):
    naive, ht, haj, ess, wmax = run(s)
    print(f"{s:<16}{naive:>8.3f}{ht:>8.3f}{haj:>8.3f}{ess:>10.0f}{wmax:>9.1f}")

print("\nStrong confounding (3.0) with propensity scores clipped to [0.05, 0.95]:")
naive, ht, haj, ess, wmax = run(3.0, trunc=0.05)
print(f"  HT={ht:.3f}  Hajek={haj:.3f}  ESS={ess:.0f}  max w={wmax:.1f}")

print("\nVariance effect of truncation (200 repeats, n=3,000, strength 3.0):")
for trunc in (None, 0.05):
    ests = [run(3.0, n=3000, trunc=trunc)[2] for _ in range(200)]
    print(f"  truncation={trunc}: Hajek mean={np.mean(ests):.3f}, std={np.std(ests):.3f}")
