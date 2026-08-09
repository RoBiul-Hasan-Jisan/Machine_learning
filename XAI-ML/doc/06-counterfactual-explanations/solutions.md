# Solutions — 06 Counterfactual Explanations

1. Results vary by run (random perturbation search), but expect a handful of features among `worst concave points`, `worst area`, `worst radius`, etc. to shift, typically by single-digit-to-tens of percent.

2. Typically several features change simultaneously because the generator perturbs *all* features with Gaussian noise each iteration rather than optimizing for a minimal subset. To favor sparsity, you could add an L1 penalty term to the selection criterion (`best_distance = np.linalg.norm(candidate - instance, ord=1)`), or explicitly perturb only a random subset of features per iteration and prefer candidates that changed the fewest.

3. Some values may fall outside the realistic range of `X_train` for that feature (e.g. a value larger than any real patient's) since the perturbation only clips to `[0, instance.max() * 1.5]`, not to the empirical distribution. A better constraint: clip each modified feature to the `[min, max]` (or a percentile range) observed in `X_train` for that specific feature, or penalize the objective by KL-divergence from the training distribution as described in the "Plausible Counterfactual" formula in the theory section of `describe.md`.

4. Change `distance = np.linalg.norm(candidate - instance)` to `distance = np.linalg.norm(candidate - instance, ord=1)`. L1 distance tends to favor counterfactuals that concentrate change in fewer features (sparser), while L2 tends to spread smaller changes across more features — this mirrors the "Sparser Counterfactual" formula in the theory section of `describe.md`.

5. "Proximal" only means the counterfactual is numerically close to the original instance (small distance) — it says nothing about whether a person could actually *cause* that change in real life. A tumor's scan-derived texture/shape measurements aren't something a patient can act on, so while the counterfactual is mathematically proximal, it isn't actionable. A domain like loan approval (income, debt, credit utilization) is far more directly actionable, since applicants can genuinely take steps to change those values.
