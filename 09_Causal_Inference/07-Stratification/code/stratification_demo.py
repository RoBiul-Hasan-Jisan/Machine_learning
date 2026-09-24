"""
06. Stratification 

  1. Reproduces theory.md 2.4's worked example (binary confounder Z) at large n.
  2. Extends to a CONTINUOUS confounder, binned into strata, and shows how the
     number of bins trades off bias vs. the stability of each stratum estimate.
  3. Deliberately creates a near-empty stratum to show the practical failure
     mode connected to positivity (Module 06).


"""

import numpy as np

rng = np.random.default_rng(seed=0)


def stratified_ate(X_strata, T, Y_obs):
    """Generic stratified ATE estimator: X_strata is an array of stratum labels."""
    n = len(T)
    total = 0.0
    details = []
    for stratum in np.unique(X_strata):
        mask = X_strata == stratum
        n_s = mask.sum()
        treated = Y_obs[mask & (T == 1)]
        control = Y_obs[mask & (T == 0)]
        if len(treated) == 0 or len(control) == 0:
            details.append((stratum, n_s, len(treated), len(control), None))
            continue  # cannot estimate this stratum -- a positivity failure
        stratum_effect = treated.mean() - control.mean()
        total += (n_s / n) * stratum_effect
        details.append((stratum, n_s, len(treated), len(control), stratum_effect))
    return total, details


# --- 1. Binary confounder, large n (theory.md 2.4) ---
print("=== 1. Binary confounder (reproducing theory.md 2.4) ===")
n = 200_000
Z = rng.binomial(1, 0.5, size=n)
T = rng.binomial(1, np.where(Z == 1, 0.85, 0.15))
TRUE_EFFECT = 0.0
noise = rng.normal(0, 3, size=n)
baseline = np.where(Z == 1, 82, 74)
Y0 = baseline + noise
Y1 = Y0 + TRUE_EFFECT
Y_obs = T * Y1 + (1 - T) * Y0

naive = Y_obs[T == 1].mean() - Y_obs[T == 0].mean()
strat_est, details = stratified_ate(Z, T, Y_obs)
print(f"Naive estimate:      {naive:.3f}")
print(f"Stratified estimate: {strat_est:.3f}  (true effect = {TRUE_EFFECT})")

# --- 2. Continuous confounder, binned; compare few bins vs many bins ---
print("\n=== 2. Continuous confounder, binned into strata ===")
n = 50_000
TRUE_EFFECT = 4.0
X_cont = rng.uniform(0, 10, size=n)
prob_treated = np.clip(X_cont / 10, 0.05, 0.95)
T2 = rng.binomial(1, prob_treated)
Y0_2 = 3 * X_cont + rng.normal(0, 2, size=n)
Y1_2 = Y0_2 + TRUE_EFFECT
Y_obs_2 = T2 * Y1_2 + (1 - T2) * Y0_2

naive2 = Y_obs_2[T2 == 1].mean() - Y_obs_2[T2 == 0].mean()
print(f"Naive estimate: {naive2:.3f}  (true effect = {TRUE_EFFECT})")

for n_bins in [2, 10, 50]:
    bins = np.digitize(X_cont, bins=np.linspace(0, 10, n_bins + 1))
    est, _ = stratified_ate(bins, T2, Y_obs_2)
    print(f"  {n_bins:>3} bins -> stratified estimate: {est:.3f}")

# --- 3. Deliberately near-empty stratum: positivity failure in stratification ---
print("\n=== 3. A stratum with almost no untreated units ===")
n = 5_000
X3 = rng.uniform(0, 10, size=n)
prob_treated3 = np.clip((X3 - 3) / 4, 0.02, 0.98)
prob_treated3[X3 > 8] = 0.99  # almost everyone with X>8 is treated
T3 = rng.binomial(1, prob_treated3)
Y0_3 = 2 * X3 + rng.normal(0, 1, size=n)
Y1_3 = Y0_3 + TRUE_EFFECT
Y_obs_3 = T3 * Y1_3 + (1 - T3) * Y0_3

bins3 = np.digitize(X3, bins=np.linspace(0, 10, 6))  # 5 bins
_, details3 = stratified_ate(bins3, T3, Y_obs_3)
for stratum, n_s, n_treat, n_ctrl, effect in details3:
    effect_str = f"{effect:.3f}" if effect is not None else "UNDEFINED (empty group)"
    print(f"  stratum {stratum}: n={n_s:>4}, treated={n_treat:>4}, control={n_ctrl:>4}, "
          f"estimate={effect_str}")
print("Notice how few control units remain in the highest-X stratum -- its estimate")
print("is either missing or based on very few points, exactly the practical failure")
print("mode theory.md section 3.1 describes.")
