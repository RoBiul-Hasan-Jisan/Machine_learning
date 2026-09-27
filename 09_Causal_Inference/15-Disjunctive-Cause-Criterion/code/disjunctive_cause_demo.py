"""
15. The Disjunctive Cause Criterion 

DAG:
    Z (confounder):    Z -> T,  Z -> Y
    I (instrument):    I -> T only (no other path to Y)
    M (mediator):      T -> M -> Y   (post-treatment, a descendant of T)
    W (Y-only cause):  W -> Y only (unrelated to T)

  1. Programmatically derives the disjunctive-cause-criterion set (causes of
     T or Y, excluding descendants of T and pure instruments) and compares it
     to the minimal backdoor-criterion set.
  2. Estimates the true ATE via explicit do(.) simulation.
  3. Compares BIAS and VARIANCE (via repeated small samples) across several
     adjustment sets: {} (naive), {Z} (minimal, valid), {Z, W} (disjunctive
     cause set, valid but larger), {Z, M} (WRONG -- includes a mediator),
     {Z, I} (includes the excluded instrument, to show its specific cost).

Run with: python disjunctive_cause_demo.py
"""

import numpy as np

rng = np.random.default_rng(seed=0)

# --- Programmatic role classification (mirrors theory.md 2.1-2.4) ---
EDGES = {("Z", "T"), ("Z", "Y"), ("I", "T"), ("T", "M"), ("M", "Y"), ("W", "Y")}


def descendants(node, edges):
    result = set()
    frontier = {b for (a, b) in edges if a == node}
    while frontier:
        result |= frontier
        frontier = {b for n in frontier for (a, b) in edges if a == n} - result
    return result


def causes_of(node, edges):
    """Ancestors of `node` -- i.e., every variable with a directed path into it."""
    result = set()
    frontier = {a for (a, b) in edges if b == node}
    while frontier:
        result |= frontier
        frontier = {a for n in frontier for (a, b) in edges if b == n} - result
    return result


def remove_node(node, edges):
    return {e for e in edges if node not in e}


candidates = {"Z", "I", "M", "W"}
desc_T = descendants("T", EDGES)
causes_T = causes_of("T", EDGES)
causes_Y = causes_of("Y", EDGES)

# A pure instrument causes T but has NO remaining path to Y once T is removed
# from the graph -- i.e., its only route to Y is THROUGH T.
EDGES_NO_T = remove_node("T", EDGES)
causes_Y_without_T = causes_of("Y", EDGES_NO_T)
pure_instruments = {v for v in candidates if v in causes_T and v not in causes_Y_without_T and v not in desc_T}

disjunctive_set = (causes_T | causes_Y) & candidates
disjunctive_set = disjunctive_set - desc_T - pure_instruments

print("=== Programmatic role classification ===")
print(f"Descendants of T (excluded, post-treatment): {desc_T & candidates}")
print(f"Pure instruments (excluded):                 {pure_instruments}")
print(f"Disjunctive cause criterion adjustment set:   {disjunctive_set}")

# --- Simulate the DAG ---
def simulate(n, force_T=None):
    Z = rng.binomial(1, 0.5, size=n)
    I = rng.binomial(1, 0.5, size=n)
    W = rng.binomial(1, 0.5, size=n)
    if force_T is None:
        p_T = np.clip(0.2 + 0.3 * Z + 0.3 * I, 0, 1)
        T = rng.binomial(1, p_T)
    else:
        T = np.full(n, force_T)
    p_M = np.where(T == 1, 0.8, 0.2)
    M = rng.binomial(1, p_M)
    p_Y = np.clip(0.1 + 0.3 * T + 0.3 * Z + 0.2 * M + 0.2 * W, 0, 1)
    Y = rng.binomial(1, p_Y)
    return dict(Z=Z, I=I, W=W, T=T, M=M, Y=Y)


n_true = 500_000
data1 = simulate(n_true, force_T=1)
data0 = simulate(n_true, force_T=0)
true_ate = data1["Y"].mean() - data0["Y"].mean()
print(f"\nTrue ATE (via do(.) simulation): {true_ate:.4f}")


def stratified_estimate(data, covariate_names):
    """Generic stratified estimator, jointly stratifying on all listed covariates."""
    T, Y = data["T"], data["Y"]
    n = len(T)
    if not covariate_names:
        return Y[T == 1].mean() - Y[T == 0].mean()
    keys = list(zip(*[data[c] for c in covariate_names]))
    keys = np.array(keys, dtype=object)
    total = 0.0
    for stratum in {tuple(k) for k in keys}:
        mask = np.all([data[c] == v for c, v in zip(covariate_names, stratum)], axis=0)
        treated = Y[mask & (T == 1)]
        control = Y[mask & (T == 0)]
        if len(treated) == 0 or len(control) == 0:
            continue
        weight = mask.mean()
        total += weight * (treated.mean() - control.mean())
    return total


print("\n=== Bias comparison, large sample (n=200,000) ===")
data_obs = simulate(200_000)
for label, covs in [("{} (naive)", []), ("{Z} (minimal backdoor)", ["Z"]),
                    ("{Z, W} (disjunctive cause set)", ["Z", "W"]),
                    ("{Z, M} (WRONG: includes mediator)", ["Z", "M"]),
                    ("{Z, I} (includes excluded instrument)", ["Z", "I"])]:
    est = stratified_estimate(data_obs, covs)
    print(f"  {label:<40}: estimate = {est:.4f}  (true = {true_ate:.4f}, bias = {est - true_ate:+.4f})")

print("\n=== Variance comparison across repeated small samples (n=3,000 each, 300 repeats) ===")
n_small, n_repeats = 3000, 300
estimates = {label: [] for label in ["{Z}", "{Z, W}", "{Z, I}"]}
for _ in range(n_repeats):
    d = simulate(n_small)
    estimates["{Z}"].append(stratified_estimate(d, ["Z"]))
    estimates["{Z, W}"].append(stratified_estimate(d, ["Z", "W"]))
    estimates["{Z, I}"].append(stratified_estimate(d, ["Z", "I"]))

for label, vals in estimates.items():
    vals = np.array(vals)
    print(f"  {label:<10}: mean={vals.mean():.4f}, std={vals.std():.4f}")

print("\n{Z, W} adjusts for an extra, genuinely irrelevant-to-bias but Y-predictive")
print("covariate (a 'precision variable') -- notice its variance is comparable to or")
print("even a bit lower than {Z} alone. {Z, I} instead adjusts for the excluded")
print("instrument -- notice its variance is often higher, for no reduction in bias,")
print("exactly the cost theory.md section 2.4 warns about.")
