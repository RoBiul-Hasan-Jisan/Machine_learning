"""
04. Causal Effects: ATE, ATT, CATE 

Simulates a randomized experiment with effect heterogeneity tied to a
covariate X, then:
  1. Estimates the ATE via the simple difference-in-means estimator, with a
     standard error and approximate 95% confidence interval (theory.md 2.4).
  2. Computes the TRUE ATE, ATT, and CATE(0)/CATE(1) directly from the
     God's-eye potential-outcomes table (only possible because we simulated
     the data ourselves -- see Module 02).

Two assignment mechanisms are provided:
  - "rct":        T assigned independently of X (a true randomized experiment)
  - "selected":   people with X=1 are more likely to be treated (breaks
                  randomization on purpose, for Exercise 3)


"""

import numpy as np

rng = np.random.default_rng(seed=0)
n = 20_000


def simulate(assignment: str):
    X = rng.binomial(1, 0.5, size=n)  # a binary covariate

    # Effect heterogeneity: the treatment helps X=1 individuals more.
    base_effect = np.where(X == 1, 6.0, 2.0)
    individual_noise_in_effect = rng.normal(0, 1, size=n)
    tau_i = base_effect + individual_noise_in_effect

    Y0 = rng.normal(50, 8, size=n)
    Y1 = Y0 + tau_i

    if assignment == "rct":
        T = rng.binomial(1, 0.5, size=n)  # independent of X and of potential outcomes
    elif assignment == "selected":
        prob_treated = np.where(X == 1, 0.8, 0.2)  # X=1 people more likely to be treated
        T = rng.binomial(1, prob_treated)
    else:
        raise ValueError("assignment must be 'rct' or 'selected'")

    Y_obs = T * Y1 + (1 - T) * Y0
    return X, Y0, Y1, tau_i, T, Y_obs


def difference_in_means_with_ci(T, Y_obs):
    y1 = Y_obs[T == 1]
    y0 = Y_obs[T == 0]
    point_estimate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    ci_low, ci_high = point_estimate - 1.96 * se, point_estimate + 1.96 * se
    return point_estimate, se, (ci_low, ci_high)


print("=== True RCT (treatment independent of X) ===")
X, Y0, Y1, tau_i, T, Y_obs = simulate("rct")

point_est, se, (ci_low, ci_high) = difference_in_means_with_ci(T, Y_obs)
print(f"Difference-in-means ATE estimate: {point_est:.3f}  (SE = {se:.3f})")
print(f"Approximate 95% CI:               [{ci_low:.3f}, {ci_high:.3f}]")

true_ate = tau_i.mean()
true_att = tau_i[T == 1].mean()
cate_0 = tau_i[X == 0].mean()
cate_1 = tau_i[X == 1].mean()
print(f"\nTrue ATE (God's-eye view):  {true_ate:.3f}")
print(f"True ATT (God's-eye view):  {true_att:.3f}  (should be close to ATE under a true RCT)")
print(f"True CATE(X=0):             {cate_0:.3f}")
print(f"True CATE(X=1):             {cate_1:.3f}")

print("\n=== Confounded assignment (X=1 individuals more likely to be treated) ===")
X, Y0, Y1, tau_i, T, Y_obs = simulate("selected")

point_est_sel, se_sel, _ = difference_in_means_with_ci(T, Y_obs)
true_ate_sel = tau_i.mean()
true_att_sel = tau_i[T == 1].mean()
cate_0_sel = tau_i[X == 0].mean()
cate_1_sel = tau_i[X == 1].mean()

print(f"Naive difference-in-means estimate: {point_est_sel:.3f}  (biased for the ATE here!)")
print(f"True ATE:      {true_ate_sel:.3f}")
print(f"True ATT:      {true_att_sel:.3f}  <- notice this is much closer to the naive estimate")
print(f"True CATE(0):  {cate_0_sel:.3f}")
print(f"True CATE(1):  {cate_1_sel:.3f}  <- ATT is pulled toward this, since X=1 people are overrepresented among the treated")
