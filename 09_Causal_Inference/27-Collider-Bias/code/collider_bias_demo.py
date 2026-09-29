""" Collider bias: M-bias, Berkson's paradox, and a descendant of a collider."""
import numpy as np

rng = np.random.default_rng(0)
n = 400_000

def ols(y, *cols):
    Z = np.column_stack([np.ones(len(y)), *cols])
    return np.linalg.lstsq(Z, y, rcond=None)[0][1:]

print("=== M-bias: T <- U1 -> M <- U2 -> Y   (true effect of T on Y = 1.0) ===")
print(f"{'strength k':<12}{'naive Y~T':>12}{'adjust for M':>15}")
for k in (0.0, 0.5, 1.0, 2.0):
    U1, U2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    T = U1 + rng.normal(0, 1, n)
    M = k * U1 + k * U2 + rng.normal(0, 1, n)
    Y = 1.0 * T + U2 + rng.normal(0, 1, n)
    print(f"{k:<12}{ols(Y, T)[0]:>12.3f}{ols(Y, T, M)[0]:>15.3f}")

print("\n=== Berkson's paradox: two independent diseases, admitted if either present ===")
A = rng.binomial(1, 0.2, n); B = rng.binomial(1, 0.2, n)
admitted = (A | B).astype(bool)
print(f"corr(A,B) in the population = {np.corrcoef(A, B)[0, 1]:+.3f}")
print(f"corr(A,B) among admitted    = {np.corrcoef(A[admitted], B[admitted])[0, 1]:+.3f}")

print("\n=== Conditioning on a DESCENDANT of a collider (T, Y independent) ===")
T = rng.normal(0, 1, n); Y = rng.normal(0, 1, n)
C = T + Y + rng.normal(0, 0.5, n)
D = C + rng.normal(0, 1.0, n)            # noisy descendant of the collider
print(f"coef of T in Y~T       = {ols(Y, T)[0]:+.3f}")
print(f"coef of T in Y~T + C   = {ols(Y, T, C)[0]:+.3f}   (adjusting for the collider)")
print(f"coef of T in Y~T + D   = {ols(Y, T, D)[0]:+.3f}   (adjusting for its descendant: weaker, still biased)")
