"""
05. Causal Assumptions 

Three short, independent demonstrations:

  1. UNCONFOUNDEDNESS VIOLATED: a confounder Z affects both T and Y, but is
     never conditioned on. The naive estimate stays biased no matter how
     much data you collect (bias, not noise).

  2. UNCONFOUNDEDNESS RESTORED: the same setup, but this time we condition
     on Z (split into groups, estimate within each, then average) -- a
     preview of Module 07's stratification estimator.

  3. POSITIVITY VIOLATED: a region of covariate space where EVERY unit is
     treated, so there is no untreated comparison group at all there.


"""

import numpy as np

rng = np.random.default_rng(seed=0)
TRUE_EFFECT = 3.0


def confounded_data(n):
    Z = rng.binomial(1, 0.5, size=n)
    T = rng.binomial(1, np.where(Z == 1, 0.85, 0.15))
    noise = rng.normal(0, 2, size=n)
    baseline = np.where(Z == 1, 70, 50)  # Z affects Y directly
    Y0 = baseline + noise
    Y1 = Y0 + TRUE_EFFECT
    Y_obs = T * Y1 + (1 - T) * Y0
    return Z, T, Y_obs


print("=== 1. Unconfoundedness violated: bias does NOT shrink with more data ===")
for n in [1_000, 100_000]:
    Z, T, Y_obs = confounded_data(n)
    naive = Y_obs[T == 1].mean() - Y_obs[T == 0].mean()
    print(f"  n={n:>7}: naive estimate = {naive:.3f}  (true effect = {TRUE_EFFECT})")

print("\n=== 2. Unconfoundedness restored: condition on Z (stratify) ===")
Z, T, Y_obs = confounded_data(100_000)
estimates_by_stratum = []
for z_val in [0, 1]:
    mask = Z == z_val
    stratum_estimate = Y_obs[mask & (T == 1)].mean() - Y_obs[mask & (T == 0)].mean()
    weight = mask.mean()
    estimates_by_stratum.append((z_val, stratum_estimate, weight))
    print(f"  Within Z={z_val} (weight={weight:.2f}): effect = {stratum_estimate:.3f}")

stratified_ate = sum(est * w for _, est, w in estimates_by_stratum)
print(f"  Stratified (weighted) estimate: {stratified_ate:.3f}  (true effect = {TRUE_EFFECT})")

print("\n=== 3. Positivity violated: a covariate region with no untreated units ===")
n = 5_000
X = rng.uniform(0, 10, size=n)
# Everyone with X > 7 is treated, no exceptions -- a hard positivity violation there.
prob_treated = np.clip((X - 3) / 4, 0.05, 1.0)
prob_treated[X > 7] = 1.0
T = rng.binomial(1, prob_treated)
Y0 = 2 * X + rng.normal(0, 1, size=n)
Y1 = Y0 + TRUE_EFFECT
Y_obs = T * Y1 + (1 - T) * Y0

region_ok = (X > 3) & (X < 5)          # has both treated and untreated
region_bad = X > 7                      # positivity violated: all treated

n_treated_bad = T[region_bad].sum()
n_control_bad = (1 - T[region_bad]).sum()
print(f"  Region X in (3,5): {T[region_ok].sum()} treated, {(1-T[region_ok]).sum()} untreated -- overlap OK")
print(f"  Region X > 7:      {n_treated_bad} treated, {n_control_bad} untreated -- NO untreated units at all")
print("  Any 'effect estimate' for X > 7 would have to come entirely from a model's")
print("  extrapolation, not from any actual comparison -- exactly what positivity rules out.")
