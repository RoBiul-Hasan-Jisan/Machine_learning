"""
01. Introduction to Causal Effect -- code demo

Simulates the coffee / lifespan / income confounding example from theory.md
section 1.1, then shows numerically:
  1. The raw (naive) correlation between coffee and lifespan.
  2. The TRUE causal effect, known here because we are simulating the data
     ourselves and therefore know each person's Y(1) and Y(0) (impossible
     in a real study -- see the "fundamental problem of causal inference").
  3. That the two numbers are very different when there is confounding.

"""

import numpy as np

rng = np.random.default_rng(seed=0)
n = 20_000

# Z: the confounder (income/free-time group). 1 = high income/free time.
Z = rng.binomial(1, 0.5, size=n)

# Treatment assignment depends on Z: high-income people drink more coffee.
# This is what makes Z a confounder -- it affects who gets "treated".
coffee_prob = np.where(Z == 1, 0.8, 0.2)
T = rng.binomial(1, coffee_prob)  # 1 = drinks a lot of coffee, 0 = does not

# TRUE causal effect of coffee on lifespan, set by us (the simulators) to be
# exactly ZERO, so we can see whether the naive analysis correctly finds "no
# effect" or is fooled by confounding into finding a large fake effect.
TRUE_EFFECT = 0.0

# Potential outcomes: baseline lifespan depends on Z (confounder), plus the
# (here, zero) causal effect of treatment, plus individual noise.
noise = rng.normal(0, 3, size=n)
baseline = np.where(Z == 1, 82, 74)  # matches theory.md's table
Y1 = baseline + TRUE_EFFECT + noise  # potential outcome if treated
Y0 = baseline + noise                # potential outcome if not treated

# Fundamental problem of causal inference: we only get to OBSERVE one of
# these two potential outcomes per person, selected by their actual T.
Y_obs = T * Y1 + (1 - T) * Y0

# --- 1. The naive (raw) comparison: just compare observed averages ---
naive_diff = Y_obs[T == 1].mean() - Y_obs[T == 0].mean()
print(f"Naive difference in average lifespan (coffee vs. no coffee): {naive_diff:.2f} years")

# --- 2. The TRUE average treatment effect (only knowable because we simulated it) ---
true_ate = (Y1 - Y0).mean()
print(f"True average treatment effect (from Y(1) - Y(0)):           {true_ate:.2f} years")

print()
print("The naive difference is large and positive, even though the true causal")
print("effect of coffee on lifespan is exactly zero by construction. The entire")
print("naive gap is explained by Z (income/free time) driving both coffee")
print("consumption and lifespan -- exactly the confounding story in theory.md 1.1.")
