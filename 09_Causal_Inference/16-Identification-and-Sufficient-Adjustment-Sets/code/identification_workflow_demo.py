"""
16. Identification and Sufficient Adjustment Sets 

A single DAG containing all four "roles" studied across Modules 08-15 at once:

    Z (confounder):   Z -> T,  Z -> Y
    I (instrument):   I -> T only
    M (mediator):     T -> M -> Y
    C (collider):     T -> C,  Y -> C

Runs the full identification workflow:
  1. Backdoor criterion search (Module 14) -> minimal valid adjustment set(s).
  2. Disjunctive cause criterion (Module 15) -> robust, DAG-light adjustment set.
  3. Simulates data, computes the true ATE via do(.), and compares the
     estimate from EVERY set in theory.md's mistake table side by side.

Run with: python identification_workflow_demo.py
"""

import numpy as np
from itertools import chain, combinations

rng = np.random.default_rng(seed=0)

EDGES = {("Z", "T"), ("Z", "Y"), ("I", "T"), ("T", "M"), ("M", "Y"), ("T", "C"), ("Y", "C")}
CANDIDATES = {"Z", "I", "M", "C"}


# ---------- Shared graph utilities (Modules 09-15) ----------
def undirected_adjacency(edges):
    adj = {n: set() for n in {x for e in edges for x in e}}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def all_paths(start, end, edges):
    adj = undirected_adjacency(edges)
    paths = []

    def dfs(node, visited, path):
        if node == end:
            paths.append(list(path))
            return
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, visited | {neighbor}, path + [neighbor])

    dfs(start, {start}, [start])
    return paths


def descendants(node, edges):
    result = set()
    frontier = {b for (a, b) in edges if a == node}
    while frontier:
        result |= frontier
        frontier = {b for n in frontier for (a, b) in edges if a == n} - result
    return result


def causes_of(node, edges):
    result = set()
    frontier = {a for (a, b) in edges if b == node}
    while frontier:
        result |= frontier
        frontier = {a for n in frontier for (a, b) in edges if b == n} - result
    return result


def remove_node(node, edges):
    return {e for e in edges if node not in e}


def is_collider_on_path(path, i, edges):
    if i == 0 or i == len(path) - 1:
        return False
    a, m, b = path[i - 1], path[i], path[i + 1]
    return (a, m) in edges and (b, m) in edges


def path_is_blocked(path, conditioning_set, edges):
    for i in range(1, len(path) - 1):
        node = path[i]
        if is_collider_on_path(path, i, edges):
            if node not in conditioning_set and not (descendants(node, edges) & conditioning_set):
                return True
        else:
            if node in conditioning_set:
                return True
    return False


def backdoor_paths(T, Y, edges):
    paths = all_paths(T, Y, edges)
    return [p for p in paths if len(p) > 1 and (p[1], p[0]) in edges]


def satisfies_backdoor_criterion(Z_candidate, T, Y, edges):
    if Z_candidate & descendants(T, edges):
        return False
    return all(path_is_blocked(p, Z_candidate, edges) for p in backdoor_paths(T, Y, edges))


# ---------- 1. Backdoor criterion search ----------
print("=== 1. Backdoor criterion: searching subsets of {Z, I, M, C} ===")
subsets = list(chain.from_iterable(combinations(sorted(CANDIDATES), r) for r in range(len(CANDIDATES) + 1)))
valid_backdoor_sets = [set(s) for s in subsets if satisfies_backdoor_criterion(set(s), "T", "Y", EDGES)]
print(f"Valid backdoor adjustment sets found: {valid_backdoor_sets}")
minimal_backdoor = min(valid_backdoor_sets, key=len)
print(f"Smallest valid set: {minimal_backdoor}")

# ---------- 2. Disjunctive cause criterion ----------
print("\n=== 2. Disjunctive cause criterion ===")
desc_T = descendants("T", EDGES)
causes_T = causes_of("T", EDGES)
causes_Y = causes_of("Y", EDGES)
EDGES_NO_T = remove_node("T", EDGES)
causes_Y_without_T = causes_of("Y", EDGES_NO_T)
pure_instruments = {v for v in CANDIDATES if v in causes_T and v not in causes_Y_without_T and v not in desc_T}
disjunctive_set = ((causes_T | causes_Y) & CANDIDATES) - desc_T - pure_instruments
print(f"Descendants of T (excluded): {desc_T & CANDIDATES}")
print(f"Pure instruments (excluded): {pure_instruments}")
print(f"Disjunctive cause criterion set: {disjunctive_set}")

# ---------- 3. Simulate, compute true ATE, and compare every set ----------
def simulate(n, force_T=None):
    Z = rng.binomial(1, 0.5, size=n)
    I = rng.binomial(1, 0.5, size=n)
    if force_T is None:
        p_T = np.clip(0.2 + 0.3 * Z + 0.3 * I, 0, 1)
        T = rng.binomial(1, p_T)
    else:
        T = np.full(n, force_T)
    M = rng.binomial(1, np.where(T == 1, 0.8, 0.2))
    p_Y = np.clip(0.1 + 0.35 * T + 0.3 * Z + 0.2 * M, 0, 1)
    Y = rng.binomial(1, p_Y)
    C = ((T == 1) | (Y == 1)).astype(int)
    return dict(Z=Z, I=I, M=M, T=T, Y=Y, C=C)


n_true = 500_000
d1 = simulate(n_true, force_T=1)
d0 = simulate(n_true, force_T=0)
true_ate = d1["Y"].mean() - d0["Y"].mean()
print(f"\n=== 3. True ATE (via do(.) simulation): {true_ate:.4f} ===")


def stratified_estimate(data, covariate_names):
    T, Y = data["T"], data["Y"]
    if not covariate_names:
        return Y[T == 1].mean() - Y[T == 0].mean()
    total = 0.0
    keys = list(zip(*[data[c] for c in covariate_names]))
    for stratum in {tuple(k) for k in keys}:
        mask = np.all([data[c] == v for c, v in zip(covariate_names, stratum)], axis=0)
        treated, control = Y[mask & (T == 1)], Y[mask & (T == 0)]
        if len(treated) == 0 or len(control) == 0:
            continue
        total += mask.mean() * (treated.mean() - control.mean())
    return total


data_obs = simulate(300_000)
sets_to_try = [
    ("{} -- naive, ignores confounder Z", []),
    (f"{sorted(minimal_backdoor)} -- backdoor criterion (minimal)", sorted(minimal_backdoor)),
    (f"{sorted(disjunctive_set)} -- disjunctive cause criterion", sorted(disjunctive_set)),
    ("['Z', 'M'] -- MISTAKE: adjusts for mediator", ["Z", "M"]),
    ("['Z', 'C'] -- MISTAKE: adjusts for collider", ["Z", "C"]),
    ("['Z', 'I'] -- adjusts for excluded instrument (no bias, but see variance)", ["Z", "I"]),
]

print(f"\n{'Adjustment set':<62}{'Estimate':>10}{'Bias':>10}")
for label, covs in sets_to_try:
    est = stratified_estimate(data_obs, covs)
    print(f"{label:<62}{est:>10.4f}{est - true_ate:>+10.4f}")
