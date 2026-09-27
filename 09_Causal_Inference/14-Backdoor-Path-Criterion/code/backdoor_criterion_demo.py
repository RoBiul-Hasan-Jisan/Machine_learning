"""
14. The Backdoor Path Criterion 

Extends Module 12's d-separation machinery into a full backdoor-criterion
checker, then:
  1. Reproduces theory.md 2.4's three candidate-set checks on the busy DAG.
  2. Brute-force searches every subset of {Z, M, C} for valid adjustment sets.
  3. Simulates data from the busy DAG and verifies the backdoor adjustment
     formula, using the one valid set found, recovers the true causal effect.

Run with: python backdoor_criterion_demo.py
"""

import numpy as np
from itertools import chain, combinations

rng = np.random.default_rng(seed=0)

EDGES = {("Z", "T"), ("Z", "Y"), ("T", "M"), ("M", "Y"), ("T", "C"), ("Y", "C")}


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
    """A backdoor path from T to Y is any path whose FIRST edge points INTO T."""
    paths = all_paths(T, Y, edges)
    result = []
    for p in paths:
        if len(p) < 2:
            continue
        first_step_into_T = (p[1], p[0]) in edges  # edge p[1] -> T
        if first_step_into_T:
            result.append(p)
    return result


def satisfies_backdoor_criterion(Z_candidate, T, Y, edges):
    """Returns (bool, reason)."""
    desc_T = descendants(T, edges)
    if Z_candidate & desc_T:
        return False, f"condition 1 FAILS: {Z_candidate & desc_T} is/are descendant(s) of {T}"
    bpaths = backdoor_paths(T, Y, edges)
    for p in bpaths:
        if not path_is_blocked(p, Z_candidate, edges):
            return False, f"condition 2 FAILS: backdoor path {p} is not blocked"
    return True, "both conditions hold"


print("=== 1. Checking three candidate adjustment sets (theory.md 2.4) ===")
for candidate in [{"Z"}, {"Z", "M"}, {"Z", "C"}]:
    ok, reason = satisfies_backdoor_criterion(candidate, "T", "Y", EDGES)
    print(f"  {candidate}: {'VALID' if ok else 'INVALID'} -- {reason}")

print("\n=== 2. Brute-force search over subsets of {Z, M, C} ===")
candidates = ["Z", "M", "C"]
subsets = list(chain.from_iterable(combinations(candidates, r) for r in range(len(candidates) + 1)))
valid_sets = []
for s in subsets:
    ok, reason = satisfies_backdoor_criterion(set(s), "T", "Y", EDGES)
    label = "{" + ", ".join(s) + "}" if s else "{}"
    print(f"  {label:<12}: {'VALID' if ok else 'invalid'}")
    if ok:
        valid_sets.append(set(s))
print(f"\nValid adjustment sets found: {valid_sets}")

print("\n=== 3. Verifying the backdoor adjustment formula numerically ===")
n = 400_000
Z = rng.binomial(1, 0.5, size=n)
T = rng.binomial(1, np.where(Z == 1, 0.7, 0.3))
M = rng.binomial(1, np.where(T == 1, 0.8, 0.2))
p_y = np.clip(0.1 + 0.3 * T + 0.3 * Z + 0.3 * M, 0, 1)
Y = rng.binomial(1, p_y)

# True effect via do(.) simulation: force T, resample M and Y downstream.
def simulate_do_T(t_value, n):
    Z_ = rng.binomial(1, 0.5, size=n)
    T_ = np.full(n, t_value)
    M_ = rng.binomial(1, np.where(T_ == 1, 0.8, 0.2))
    p_y_ = np.clip(0.1 + 0.3 * T_ + 0.3 * Z_ + 0.3 * M_, 0, 1)
    Y_ = rng.binomial(1, p_y_)
    return Y_.mean()

true_ate = simulate_do_T(1, n) - simulate_do_T(0, n)

# Backdoor adjustment formula using the one valid set found, {Z}:
def backdoor_estimate(t_value):
    total = 0.0
    for z_val in [0, 1]:
        p_z = np.mean(Z == z_val)
        mask = (T == t_value) & (Z == z_val)
        p_y_given_tz = Y[mask].mean()
        total += p_y_given_tz * p_z
    return total

backdoor_ate = backdoor_estimate(1) - backdoor_estimate(0)

print(f"True ATE (via explicit do(.) simulation):      {true_ate:.4f}")
print(f"Backdoor-adjustment-formula estimate (Z only): {backdoor_ate:.4f}")
print("These should match closely -- confirming {Z} is a valid adjustment set in practice,")
print("not just according to the graphical criterion.")
