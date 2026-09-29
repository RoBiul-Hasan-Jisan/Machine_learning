""" two models, same observed data, different causal effect."""

import numpy as np

rng = np.random.default_rng(0)
n = 1_000_000

def model_A(n, do_T=None):
    T = rng.normal(0, 1, n) if do_T is None else np.full(n, float(do_T))
    Y = 0.6 * T + rng.normal(0, np.sqrt(2), n)
    return T, Y

def model_B(n, do_T=None):
    U = rng.normal(0, 1, n)
    T = 0.5 * U + rng.normal(0, np.sqrt(0.75), n) if do_T is None else np.full(n, float(do_T))
    Y = 1.2 * U + rng.normal(0, np.sqrt(0.92), n)   # T does not appear in Y's equation
    return T, Y

for name, model in [("A (no confounding)", model_A), ("B (pure confounding)", model_B)]:
    T, Y = model(n)
    cov = np.cov(T, Y)
    _, Y1 = model(n, do_T=1)
    _, Y0 = model(n, do_T=0)
    print(f"Model {name}")
    print(f"  Observed cov matrix [[Var T, Cov],[Cov, Var Y]] = "
          f"[[{cov[0,0]:.3f}, {cov[0,1]:.3f}],[{cov[1,0]:.3f}, {cov[1,1]:.3f}]]")
    print(f"  Observed regression slope of Y on T: {cov[0,1]/cov[0,0]:.3f}")
    print(f"  TRUE causal effect E[Y|do(T=1)]-E[Y|do(T=0)]: {Y1.mean()-Y0.mean():.3f}\n")

print("Same observed distribution, different causal effect -> NOT identified without")
print("extra assumptions (e.g. measuring U and adjusting for it).")

# If U WERE measured, the effect becomes identified: regress Y on (T, U).
U = rng.normal(0, 1, n)
T = 0.5 * U + rng.normal(0, np.sqrt(0.75), n)
Y = 1.2 * U + rng.normal(0, np.sqrt(0.92), n)
X = np.column_stack([np.ones(n), T, U])
beta = np.linalg.lstsq(X, Y, rcond=None)[0]
print(f"\nModel B with U observed: coefficient on T after adjusting for U = {beta[1]:.3f} (true effect 0)")
