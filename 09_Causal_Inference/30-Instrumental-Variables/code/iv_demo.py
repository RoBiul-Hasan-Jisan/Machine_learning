""" Instrumental variables: 2SLS with strong/weak instruments; LATE with compliers."""
import numpy as np

rng = np.random.default_rng(0)

def cov(a, b): return np.cov(a, b)[0, 1]

def one_draw(n, pi):
    Z = rng.normal(0, 1, n)
    U = rng.normal(0, 1, n)
    T = pi * Z + U + rng.normal(0, 1, n)
    Y = 1.0 * T + 2.0 * U + rng.normal(0, 1, n)          # true beta = 1
    ols = cov(T, Y) / np.var(T, ddof=1)
    iv = cov(Z, Y) / cov(Z, T)
    r = np.corrcoef(Z, T)[0, 1] ** 2
    F = r / (1 - r) * (n - 2)
    return ols, iv, F

print("=== Continuous treatment (true effect = 1.0) ===")
print(f"{'pi':<8}{'n':<8}{'OLS':>8}{'IV mean':>10}{'IV std':>9}{'IV median':>11}{'mean F':>9}")
for pi, n in ((1.0, 2000), (0.05, 2000)):
    draws = np.array([one_draw(n, pi) for _ in range(500)])
    print(f"{pi:<8}{n:<8}{draws[:, 0].mean():>8.3f}{draws[:, 1].mean():>10.3f}{draws[:, 1].std():>9.3f}"
          f"{np.median(draws[:, 1]):>11.3f}{draws[:, 2].mean():>9.1f}")

print("\n=== Binary instrument: LATE for compliers ===")
n = 1_000_000
typ = rng.choice(["always", "never", "complier"], size=n, p=[0.2, 0.3, 0.5])
Z = rng.binomial(1, 0.5, n)
T = np.where(typ == "always", 1, np.where(typ == "never", 0, Z))
tau = np.where(typ == "always", 4.0, np.where(typ == "never", 0.0, 2.0))
base = np.where(typ == "always", 2.0, np.where(typ == "never", -1.0, 0.0))   # confounding via type
Y0 = base + rng.normal(0, 1, n)
Y = np.where(T == 1, Y0 + tau, Y0)
wald = (Y[Z == 1].mean() - Y[Z == 0].mean()) / (T[Z == 1].mean() - T[Z == 0].mean())
print(f"True ATE (all units)        = {tau.mean():.3f}")
print(f"True LATE (compliers)       = 2.000")
print(f"Naive treated - untreated   = {Y[T == 1].mean() - Y[T == 0].mean():.3f}")
print(f"Wald / IV estimate          = {wald:.3f}   <- recovers the complier effect, not the ATE")
