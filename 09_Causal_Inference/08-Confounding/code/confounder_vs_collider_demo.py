"""
08. Confounding 

Two side-by-side simulations:

  1. CONFOUNDER: Z -> T, Z -> Y, no direct T -> Y effect.
     Naive estimate is biased; adjusting for Z (stratifying, as in Module 07)
     removes the bias.

  2. COLLIDER: T and Y are independent; both cause C.
     The unconditional estimate correctly finds ~0 association; conditioning
     on C (restricting to C=1) introduces a spurious one.

Run with: python confounder_vs_collider_demo.py
"""

import numpy as np

rng = np.random.default_rng(seed=0)
n = 100_000

print("=== 1. Confounder case: Z -> T, Z -> Y, T has NO true effect on Y ===")
Z = rng.binomial(1, 0.5, size=n)
T = rng.binomial(1, np.where(Z == 1, 0.8, 0.2))
Y = np.where(Z == 1, 80, 60) + rng.normal(0, 3, size=n)  # Y depends on Z only

naive = Y[T == 1].mean() - Y[T == 0].mean()
adjusted = sum(
    (np.mean(Z == z)) * (Y[(T == 1) & (Z == z)].mean() - Y[(T == 0) & (Z == z)].mean())
    for z in [0, 1]
)
print(f"Naive (unadjusted) estimate:         {naive:.3f}  (true effect = 0)")
print(f"Adjusted (stratified on Z) estimate: {adjusted:.3f}  (true effect = 0)")

print("\n=== 2. Collider case: T and Y independent; both cause C ===")
T2 = rng.binomial(1, 0.5, size=n)
Y2 = rng.binomial(1, 0.5, size=n)  # genuinely independent of T2
# C = 1 whenever EITHER T2 or Y2 is 1 (a simple deterministic "OR" collider)
C = ((T2 == 1) | (Y2 == 1)).astype(int)

unconditional_corr = np.corrcoef(T2, Y2)[0, 1]
mask_c1 = C == 1
conditional_corr = np.corrcoef(T2[mask_c1], Y2[mask_c1])[0, 1]

print(f"Unconditional correlation(T, Y):        {unconditional_corr:.3f}  (true relationship: independent)")
print(f"Correlation(T, Y), CONDITIONED on C=1:  {conditional_corr:.3f}  (spurious negative association appears)")
print("\nConditioning on the collider C manufactured a negative association between")
print("T and Y that does not exist in the full population -- exactly section 2.4's warning.")
