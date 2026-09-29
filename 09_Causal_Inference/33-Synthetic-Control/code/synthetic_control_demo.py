""" Synthetic control: constrained weights, gap estimate, placebo-in-space test."""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)
J, TT, T0, EFFECT = 30, 40, 25, 3.0

F = np.cumsum(rng.normal(0, 1, (TT, 3)), axis=0)               # 3 latent factors over time
delta = 0.2 * np.arange(TT)
mu = rng.normal(0, 1, (J, 3))                                  # donor loadings
Y_donors = delta[:, None] + F @ mu.T + rng.normal(0, 0.3, (TT, J))
w_true = np.zeros(J); w_true[[3, 7, 12, 20]] = [0.4, 0.3, 0.2, 0.1]
y1_untreated = delta + F @ (mu.T @ w_true) + rng.normal(0, 0.3, TT)
y1 = y1_untreated.copy(); y1[T0:] += EFFECT

def fit_weights(y_pre, D_pre):
    k = D_pre.shape[1]
    obj = lambda w: np.sum((y_pre - D_pre @ w) ** 2)
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x

def gap_stats(y, D):
    w = fit_weights(y[:T0], D[:T0])
    gap = y - D @ w
    pre = np.sqrt(np.mean(gap[:T0] ** 2)); post = np.sqrt(np.mean(gap[T0:] ** 2))
    return w, gap, pre, post

w, gap, pre, post = gap_stats(y1, Y_donors)
print(f"Pre-treatment RMSPE = {pre:.3f}   (noise sd is 0.3)")
print(f"Largest donor weights: " + ", ".join(f"unit {j}: {w[j]:.2f}" for j in np.argsort(-w)[:4]))
print(f"True effect = {EFFECT:.2f};  estimated average post gap = {gap[T0:].mean():.3f}")
print(f"Naive treated post-pre change = {y1[T0:].mean() - y1[:T0].mean():.3f}   <- includes the time trend")

# Placebo in space: pretend each donor is treated
ratios = []
for j in range(J):
    others = np.delete(np.arange(J), j)
    _, _, p_pre, p_post = gap_stats(Y_donors[:, j], Y_donors[:, others])
    ratios.append(p_post / p_pre)
treated_ratio = post / pre
rank = 1 + sum(r >= treated_ratio for r in ratios)
print(f"\nPost/pre RMSPE ratio: treated = {treated_ratio:.1f}, placebo median = {np.median(ratios):.1f}, max = {max(ratios):.1f}")
print(f"Placebo p-value = {rank}/{J + 1} = {rank / (J + 1):.3f}")
