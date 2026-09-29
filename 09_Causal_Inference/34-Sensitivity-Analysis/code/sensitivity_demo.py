""" Sensitivity analysis: E-value, omitted variable bias, robustness value."""
import numpy as np

def e_value(rr):
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))

print("=== (i) E-values ===")
rr, lo, hi = 1.8, 1.3, 2.5
print(f"RR = {rr} (95% CI {lo}-{hi}): E-value (point) = {e_value(rr):.2f}, E-value (CI limit nearest 1) = {e_value(lo):.2f}")
print(f"RR = 0.6: E-value = {e_value(0.6):.2f}")
print(f"RR = 1.1: E-value = {e_value(1.1):.2f}   (weak association is easy to explain away)")

print("\n=== (ii) Omitted variable bias in a linear model ===")
rng = np.random.default_rng(0)
n = 500_000
tau, gamma, d = 1.0, 2.0, 0.6
U = rng.normal(0, 1, n)
T = d * U + rng.normal(0, 1, n)
Y = tau * T + gamma * U + rng.normal(0, 1, n)
short = np.cov(T, Y)[0, 1] / np.var(T, ddof=1)
delta = np.cov(T, U)[0, 1] / np.var(T, ddof=1)
print(f"true tau = {tau}; short-regression estimate = {short:.3f}")
print(f"formula: tau + gamma*delta = {tau} + {gamma}*{delta:.3f} = {tau + gamma * delta:.3f}")

print("\n=== (iii) Robustness value (Cinelli-Hazlett) ===")
X = rng.normal(0, 1, n)
T2 = 0.5 * X + rng.normal(0, 1, n)
Y2 = 0.4 * T2 + 0.8 * X + rng.normal(0, 1, n)
D = np.column_stack([np.ones(n), T2, X])
beta = np.linalg.lstsq(D, Y2, rcond=None)[0]
resid = Y2 - D @ beta
df = n - D.shape[1]
se = np.sqrt(resid @ resid / df * np.linalg.inv(D.T @ D)[1, 1])
t = beta[1] / se
f = abs(t) / np.sqrt(df)
rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
print(f"coef on T = {beta[1]:.3f}, t = {t:.1f}, partial Cohen's f = {f:.3f}")
print(f"Robustness value RV = {rv:.3f}: a confounder explaining >= {rv:.1%} of the residual variance of both T and Y could nullify the estimate")
# Benchmark against the observed covariate X (partial R^2 with T and with Y)
r2_xt = np.corrcoef(X, T2)[0, 1] ** 2
y_net_of_T = Y2 - beta[1] * T2                     # outcome with T's contribution removed
r2_xy = 1 - np.var(resid) / np.var(y_net_of_T)      # share of that remaining variance explained by X
print(f"Benchmark: observed covariate X explains {r2_xt:.1%} of T and ~{r2_xy:.1%} of Y (given T)")
