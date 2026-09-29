""" Mediation: product of coefficients, with and without M-Y confounding."""
import numpy as np

rng = np.random.default_rng(0)
n = 500_000
alpha, b, c = 0.8, 1.5, 0.5          # T->M, M->Y, T->Y

def ols(y, *cols):
    Z = np.column_stack([np.ones(len(y)), *cols])
    return np.linalg.lstsq(Z, y, rcond=None)[0][1:]

def run(conf):
    T = rng.binomial(1, 0.5, n).astype(float)               # randomized
    U = rng.normal(0, 1, n)
    M = alpha * T + conf * U + rng.normal(0, 1, n)
    Y = c * T + b * M + conf * U + rng.normal(0, 1, n)
    a_hat = ols(M, T)[0]
    c_hat, b_hat = ols(Y, T, M)
    total = ols(Y, T)[0]
    return a_hat, b_hat, c_hat, total

print("Truth: TE = 1.700, NDE = 0.500, NIE = 1.200\n")
for label, conf in (("No M-Y confounder", 0.0), ("Unmeasured M-Y confounder U (strength 1)", 1.0)):
    a_hat, b_hat, c_hat, total = run(conf)
    print(label)
    print(f"  total effect (Y~T)        = {total:.3f}   <- still unbiased (T randomized)")
    print(f"  NDE  = c_hat              = {c_hat:.3f}")
    print(f"  NIE  = alpha_hat * b_hat  = {a_hat * b_hat:.3f}")
    print(f"  TE decomposed (NDE+NIE)   = {c_hat + a_hat * b_hat:.3f}\n")
print("Adjusting for M when you want the TOTAL effect (no confounder case) gives only the direct part:")
T = rng.binomial(1, 0.5, n).astype(float); M = alpha * T + rng.normal(0, 1, n)
Y = c * T + b * M + rng.normal(0, 1, n)
print(f"  Y ~ T + M, coefficient on T = {ols(Y, T, M)[0]:.3f}  (total effect is {c + alpha * b:.3f})")
