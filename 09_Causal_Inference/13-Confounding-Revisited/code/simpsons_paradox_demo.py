"""
13. Confounding Revisited 

Reproduces the classic kidney-stone-treatment Simpson's paradox
(Julious & Mullee, 1994; also widely cited in causal inference textbooks):

    Treatment A beats Treatment B within EVERY stone-size subgroup,
    but Treatment B appears to beat Treatment A overall.

Z = stone size (small / large) is the confounder: it affects both which
treatment a patient tends to receive (doctors preferentially gave A to
easier, small-stone cases) and the baseline success rate (small stones are
just easier to treat successfully, regardless of treatment).

Run with: python simpsons_paradox_demo.py
"""

import numpy as np

# Real counts from the classic dataset: (successes, total) for each (treatment, stone size)
counts = {
    ("A", "small"): (81, 87),
    ("B", "small"): (234, 270),
    ("A", "large"): (192, 263),
    ("B", "large"): (55, 80),
}

# --- Build individual-level arrays from the counts (deterministic, matching the real data exactly) ---
T, Z, Y = [], [], []
for (treatment, stone_size), (successes, total) in counts.items():
    T += [treatment] * total
    Z += [stone_size] * total
    Y += [1] * successes + [0] * (total - successes)
T, Z, Y = np.array(T), np.array(Z), np.array(Y)

print("=== Success rates within each stone-size stratum ===")
stratum_diffs = {}
stratum_weights = {}
for stone_size in ["small", "large"]:
    mask = Z == stone_size
    rate_a = Y[mask & (T == "A")].mean()
    rate_b = Y[mask & (T == "B")].mean()
    diff = rate_a - rate_b
    weight = mask.mean()
    stratum_diffs[stone_size] = diff
    stratum_weights[stone_size] = weight
    print(f"  {stone_size:>5} stones: A success rate = {rate_a:.3f}, B success rate = {rate_b:.3f}, "
          f"diff (A - B) = {diff:+.3f}  ({'A better' if diff > 0 else 'B better'})")

print("\n=== Marginal (unadjusted, pooled) comparison ===")
overall_rate_a = Y[T == "A"].mean()
overall_rate_b = Y[T == "B"].mean()
overall_diff = overall_rate_a - overall_rate_b
print(f"  Overall: A success rate = {overall_rate_a:.3f}, B success rate = {overall_rate_b:.3f}, "
      f"diff (A - B) = {overall_diff:+.3f}  ({'A better' if overall_diff > 0 else 'B better'})")
print("  *** This CONTRADICTS both stratum-level results above -- Simpson's paradox. ***")

print("\n=== Stratified (adjusted for Z = stone size) estimate ===")
stratified_diff = sum(stratum_weights[s] * stratum_diffs[s] for s in stratum_diffs)
print(f"  Stratified diff (A - B), weighted by stratum size: {stratified_diff:+.3f}  "
      f"({'A better' if stratified_diff > 0 else 'B better'})")
print("  This matches the direction found WITHIN every stratum -- adjusting for Z blocks")
print("  the open backdoor path T <- Z -> Y (doctors' preference for giving A to easier")
print("  small-stone cases), exactly as theory.md section 2.1 predicts.")

print("\n=== Why the confounding happened: treatment assignment differed sharply by stratum ===")
for stone_size in ["small", "large"]:
    mask = Z == stone_size
    frac_a = (T[mask] == "A").mean()
    print(f"  Among {stone_size}-stone patients: {frac_a:.1%} received Treatment A")
print("  Treatment A was given disproportionately to the EASIER (small-stone) cases,")
print("  which inflates B's apparent overall performance once the groups are pooled.")
