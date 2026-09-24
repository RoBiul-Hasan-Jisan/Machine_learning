"""
02. Potential Outcomes and Counterfactuals 

Builds a "God's-eye view" potential-outcomes table (theory.md section 2.3) for
a simulated population, then compares:
  - the TRUE ATE, computable only because we generated Y(0) and Y(1) ourselves
  - the NAIVE observed-data estimate, which is all a real analyst ever sees

Two treatment-assignment mechanisms are provided so you can reproduce
Exercise 3 from theory.md:
  - "random":   T_i assigned independently of potential outcomes (fair coin)
  - "selected": people with higher Y(0) are more likely to get T_i = 1

"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=0)
n = 10_000
TRUE_EFFECT = 5.0  # same true effect for everyone, for simplicity in this demo


def simulate(assignment: str):
    # Potential outcomes: Y(0) varies across people; Y(1) = Y(0) + true effect + a
    # little extra individual-level variation in how much the treatment helps.
    Y0 = rng.normal(60, 10, size=n)
    individual_variation = rng.normal(0, 2, size=n)
    Y1 = Y0 + TRUE_EFFECT + individual_variation

    if assignment == "random":
        T = rng.binomial(1, 0.5, size=n)
    elif assignment == "selected":
        # People with higher Y(0) are more likely to be treated -- this creates
        # exactly the kind of confounding/selection Exercise 2 asks you to spot.
        prob_treated = 1 / (1 + np.exp(-(Y0 - 60) / 5))
        T = rng.binomial(1, prob_treated)
    else:
        raise ValueError("assignment must be 'random' or 'selected'")

    Y_obs = T * Y1 + (1 - T) * Y0

    df = pd.DataFrame({"Y0": Y0, "Y1": Y1, "tau_i": Y1 - Y0, "T": T, "Y_obs": Y_obs})
    return df


def report(df, label):
    true_ate = df["tau_i"].mean()
    naive_diff = df.loc[df["T"] == 1, "Y_obs"].mean() - df.loc[df["T"] == 0, "Y_obs"].mean()
    print(f"\n--- Assignment mechanism: {label} ---")
    print(f"True ATE (God's-eye view, uses Y0 and Y1 directly): {true_ate:.3f}")
    print(f"Naive observed-data difference in means:            {naive_diff:.3f}")
    print(f"Gap (naive - true):                                 {naive_diff - true_ate:.3f}")


for mechanism in ["random", "selected"]:
    df = simulate(mechanism)
    report(df, mechanism)

print("\nPreview of the full God's-eye table (first 8 rows, 'selected' assignment).")
print("In a real study you would only ever see the T and Y_obs columns:")
print(simulate("selected").head(8).round(2).to_string(index=True))
