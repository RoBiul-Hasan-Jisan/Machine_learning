""" Sharp regression discontinuity with local linear regression."""
import numpy as np

rng = np.random.default_rng(0)
n = 20_000
X = rng.uniform(-1, 1, n)
T = (X >= 0).astype(float)
TAU = 2.0
Y = 0.5 + 1.5 * X + 2.0 * X ** 2 + TAU * T + rng.normal(0, 1, n)

def local_linear(x, y, c, h):
    def side_intercept(mask):
        u = x[mask] - c
        w = np.clip(1 - np.abs(u) / h, 0, None)           # triangular kernel
        D = np.column_stack([np.ones(mask.sum()), u])
        W = w[:, None]
        beta = np.linalg.solve(D.T @ (W * D), D.T @ (w * y[mask]))
        return beta[0]
    inside = np.abs(x - c) < h
    return side_intercept(inside & (x >= c)) - side_intercept(inside & (x < c))

print(f"True jump at the cutoff = {TAU:.3f}\n")
print(f"Naive treated - control (all data)      = {Y[T == 1].mean() - Y[T == 0].mean():.3f}")
print("\nLocal linear RD by bandwidth:")
for h in (0.05, 0.1, 0.25, 0.5, 1.0):
    print(f"  h = {h:<5}: {local_linear(X, Y, 0.0, h):.3f}")
print("\nPlacebo cutoffs (no treatment change there; should be ~0):")
for c in (-0.5, 0.5):
    print(f"  cutoff {c:+.1f}: {local_linear(X, Y, c, 0.25):+.3f}")
