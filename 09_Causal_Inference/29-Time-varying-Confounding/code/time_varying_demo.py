""" Time-varying confounding: naive, regression adjustment, g-formula, IPW."""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-z))

def gen(n, a0=None, a1=None):
    U = rng.normal(0, 1, n)
    A0 = rng.binomial(1, 0.5, n) if a0 is None else np.full(n, a0)
    L1 = rng.binomial(1, sig(-0.2 + 1.2 * U - 3.0 * A0))
    A1 = rng.binomial(1, sig(-0.5 + 3.0 * L1 + 0.5 * A0)) if a1 is None else np.full(n, a1)
    Y = 1.0 * A0 + 1.0 * A1 - 3.0 * L1 + 1.0 * U + rng.normal(0, 1, n)
    return A0, L1, A1, Y

# Truth: simulate the intervention "always treat" vs "never treat"
big = 2_000_000
truth = gen(big, 1, 1)[3].mean() - gen(big, 0, 0)[3].mean()
print(f"TRUE effect of always- vs never-treat = {truth:.3f}\n")

n = 300_000
A0, L1, A1, Y = gen(n)

# Naive: compare always vs never treated, unadjusted
al, nv = (A0 == 1) & (A1 == 1), (A0 == 0) & (A1 == 0)
print(f"Naive (always vs never, unadjusted)      = {Y[al].mean() - Y[nv].mean():.3f}")

# Regression adjusting for L1: sum of coefficients on A0 and A1
Z = np.column_stack([np.ones(n), A0, A1, L1])
b = np.linalg.lstsq(Z, Y, rcond=None)[0]
print(f"Regression adjusting for L1 (b_A0+b_A1)  = {b[1] + b[2]:.3f}")

# Parametric g-formula (saturated outcome model over the binary history)
def cell_mean(a0, l, a1):
    m = (A0 == a0) & (L1 == l) & (A1 == a1)
    return Y[m].mean()
def gformula(a0, a1):
    p_l1 = L1[A0 == a0].mean()
    return p_l1 * cell_mean(a0, 1, a1) + (1 - p_l1) * cell_mean(a0, 0, a1)
print(f"G-formula                                = {gformula(1, 1) - gformula(0, 0):.3f}")

# IPW with stabilized-style weights: 1 / (P(A0) * P(A1 | A0, L1))
F = np.column_stack([A0, L1, A0 * L1])
pA1 = LogisticRegression(C=1e6, max_iter=1000).fit(F, A1).predict_proba(F)[:, 1]
pA1_obs = np.where(A1 == 1, pA1, 1 - pA1)
pA0_obs = np.where(A0 == 1, A0.mean(), 1 - A0.mean())
w = 1 / (pA0_obs * pA1_obs)
ipw = np.sum(w[al] * Y[al]) / np.sum(w[al]) - np.sum(w[nv] * Y[nv]) / np.sum(w[nv])
print(f"IPW (marginal structural, Hajek)         = {ipw:.3f}")
