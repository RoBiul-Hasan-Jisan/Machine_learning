"""
12. Conditional Independence and d-Separation 

Builds a general d_separated(x, y, conditioning_set, edges) function by
enumerating every path between x and y (ignoring direction) and checking
whether ALL of them are blocked, using the per-path rule from Module 11.

Applies it to:
  1. theory.md 2.2's two queries on the "busy" DAG.
  2. A numerical check that graphical d-separation matches statistical
     conditional independence on simulated data.
  3. A sweep over candidate conditioning sets for a chosen variable pair.

Run with: python d_separation_demo.py
"""

import numpy as np
from itertools import chain, combinations

rng = np.random.default_rng(seed=0)

EDGES = {("Z", "T"), ("Z", "Y"), ("T", "M"), ("M", "Y"), ("T", "C"), ("Y", "C")}
NODES = {n for e in EDGES for n in e}


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


def d_separated(x, y, conditioning_set, edges, exclude_direct_edge=False):
    paths = all_paths(x, y, edges)
    if exclude_direct_edge:
        paths = [p for p in paths if len(p) > 2]
    return all(path_is_blocked(p, conditioning_set, edges) for p in paths)


print("=== 1. Graphical queries on the busy DAG (theory.md 2.2) ===")
print(f"T d-sep Y given {{Z}}?          {d_separated('T', 'Y', {'Z'}, EDGES)}")
print(f"T d-sep Y given {{Z, M}}?       {d_separated('T', 'Y', {'Z', 'M'}, EDGES)}")
print(f"T d-sep Y given {{Z, M}}, "
      f"excluding the direct T->Y edge? {d_separated('T', 'Y', {'Z', 'M'}, EDGES, exclude_direct_edge=True)}")

print("\n=== 2. Numerical check: does statistical independence match the graphical answer? ===")
n = 300_000
Z = rng.normal(0, 1, n)
T = Z + rng.normal(0, 1, n)
M = T + rng.normal(0, 1, n)
Y = 0.5 * T + M + Z + rng.normal(0, 1, n)  # T affects Y directly AND via M; Z confounds T,Y
C = T + Y + rng.normal(0, 1, n)  # collider, not used in these two queries

def corr_given_slice(a, b, slice_vars, tol=0.05):
    mask = np.ones(len(a), dtype=bool)
    for v in slice_vars:
        mask &= np.abs(v - np.median(v)) < tol
    return np.corrcoef(a[mask], b[mask])[0, 1], mask.sum()

corr_given_z, n1 = corr_given_slice(T, Y, [Z])
print(f"corr(T,Y | Z~fixed) = {corr_given_z:.3f} on n={n1} points "
      "(nonzero expected: direct effect + mediator path still open)")

corr_given_zm, n2 = corr_given_slice(T, Y, [Z, M])
print(f"corr(T,Y | Z~fixed, M~fixed) = {corr_given_zm:.3f} on n={n2} points")
print("(NOT expected to be ~0 here: T has a direct edge straight to Y, coefficient 0.5,")
print(" so this remaining correlation reflects exactly that direct effect, with the")
print(" fork (via Z) and the mediated path (via M) both now blocked. d-separation only")
print(" claims non-causal/mediating association vanishes -- a genuine direct causal")
print(" edge is never something d-separation blocks or removes.)")

print("\n=== 3. Sweep over candidate conditioning sets for T, Y ===")
candidates = ["Z", "M", "C"]
all_subsets = list(chain.from_iterable(combinations(candidates, r) for r in range(len(candidates) + 1)))
print(f"{'Conditioning set':<15}{'d-separated? (excluding direct edge)':>40}")
for subset in all_subsets:
    result = d_separated("T", "Y", set(subset), EDGES, exclude_direct_edge=True)
    label = "{" + ", ".join(subset) + "}" if subset else "{}"
    print(f"{label:<15}{str(result):>40}")
