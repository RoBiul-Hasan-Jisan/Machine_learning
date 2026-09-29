""" propensity score matching with replacement and a caliper (ATT)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
n = 20_000
X = rng.normal(0, 1, (n, 3))
T = rng.binomial(1, 1 / (1 + np.exp(-(0.9 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2] - 0.5))))
Y = 2 * T + 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] + rng.normal(0, 1, n)   # ATT = ATE = 2

e = LogisticRegression(C=1e6, max_iter=1000).fit(X, T).predict_proba(X)[:, 1]
logit = np.log(e / (1 - e))
caliper = 0.2 * logit.std()

treated = np.where(T == 1)[0]; control = np.where(T == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(logit[control, None])
dist, idx = nn.kneighbors(logit[treated, None])
ok = dist[:, 0] <= caliper
mt = treated[ok]; mc = control[idx[ok, 0]]

def smd(x, a, b):
    return (x[a].mean() - x[b].mean()) / np.sqrt((x[a].var() + x[b].var()) / 2)

print(f"Treated: {len(treated)}, matched within caliper: {ok.sum()}, distinct controls used: {len(set(mc))}")
print(f"{'cov':<5}{'SMD before':>12}{'SMD after':>12}")
for j in range(3):
    print(f"X{j:<4}{smd(X[:, j], treated, control):>12.3f}{smd(X[:, j], mt, mc):>12.3f}")
print(f"\nNaive difference = {Y[treated].mean() - Y[control].mean():.3f}")
print(f"Matched ATT      = {np.mean(Y[mt] - Y[mc]):.3f}   (true 2.000)")
