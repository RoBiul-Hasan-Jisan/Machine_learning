"""
11. Paths and Associations 

  1. Simulates a pure chain, a pure fork, and a pure collider, and checks
     (via simple correlation) that association appears/disappears exactly
     where theory.md sections 2.1-2.3 predict, with and without conditioning
     on the middle node.

  2. Implements the general path-blocking rule (theory.md 2.4) and applies it
     to the "busy" 4-path DAG from theory.md 2.5, reproducing that section's
     table programmatically.

  3. Extends the collider path with a descendant D and shows conditioning on
     D partially opens the path too.

Run with: python path_blocking_demo.py
"""

import numpy as np

rng = np.random.default_rng(seed=0)
n = 200_000


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


print("=== 1. Chain: T -> M -> Y ===")
T = rng.normal(0, 1, size=n)
M = T + rng.normal(0, 1, size=n)
Y = M + rng.normal(0, 1, size=n)
print(f"corr(T, Y), unconditional:         {corr(T, Y):.3f}  (should be nonzero -- open)")
# "Condition on M" approximately, by looking within a narrow slice of M
mask = np.abs(M - np.median(M)) < 0.05
print(f"corr(T, Y), conditioned on M~fixed: {corr(T[mask], Y[mask]):.3f}  (should be ~0 -- blocked)")

print("\n=== 2. Fork: T <- M -> Y ===")
M2 = rng.normal(0, 1, size=n)
T2 = M2 + rng.normal(0, 1, size=n)
Y2 = M2 + rng.normal(0, 1, size=n)
print(f"corr(T, Y), unconditional:         {corr(T2, Y2):.3f}  (should be nonzero -- open)")
mask2 = np.abs(M2 - np.median(M2)) < 0.05
print(f"corr(T, Y), conditioned on M~fixed: {corr(T2[mask2], Y2[mask2]):.3f}  (should be ~0 -- blocked)")

print("\n=== 3. Collider: T -> M <- Y ===")
T3 = rng.normal(0, 1, size=n)
Y3 = rng.normal(0, 1, size=n)  # independent of T3
M3 = T3 + Y3 + rng.normal(0, 1, size=n)
print(f"corr(T, Y), unconditional:         {corr(T3, Y3):.3f}  (should be ~0 -- blocked)")
mask3 = np.abs(M3 - np.median(M3)) < 0.05
print(f"corr(T, Y), conditioned on M~fixed: {corr(T3[mask3], Y3[mask3]):.3f}  (should be nonzero -- OPENED)")


# --- Path-blocking rule, applied to the "busy" DAG from theory.md 2.5 ---
print("\n=== 4. General path-blocking rule on the busy 4-path DAG ===")

EDGES = {("Z", "T"), ("Z", "Y"), ("T", "M"), ("M", "Y"), ("T", "C"), ("Y", "C")}

def is_collider_on_path(path, i):
    """Node path[i] is a collider along this path if both neighbors point INTO it."""
    if i == 0 or i == len(path) - 1:
        return False
    a, m, b = path[i - 1], path[i], path[i + 1]
    return (a, m) in EDGES and (b, m) in EDGES


def descendants(node):
    result = set()
    frontier = {b for (a, b) in EDGES if a == node}
    while frontier:
        result |= frontier
        frontier = {b for n in frontier for (a, b) in EDGES if a == n} - result
    return result


def is_blocked(path, conditioning_set):
    for i in range(1, len(path) - 1):
        node = path[i]
        if is_collider_on_path(path, i):
            # Collider: blocked UNLESS the collider or a descendant is conditioned on.
            if node not in conditioning_set and not (descendants(node) & conditioning_set):
                return True  # blocked at this collider
        else:
            # Chain or fork: blocked IF the node is conditioned on.
            if node in conditioning_set:
                return True
    return False


paths = {
    "T -> Y (direct)": ["T", "Y"],
    "T <- Z -> Y (fork)": ["T", "Z", "Y"],
    "T -> M -> Y (chain)": ["T", "M", "Y"],
    "T -> C <- Y (collider)": ["T", "C", "Y"],
}

conditioning_sets = {
    "{}": set(),
    "{Z}": {"Z"},
    "{Z, M}": {"Z", "M"},
    "{Z, C}": {"Z", "C"},
}

header = f"{'Path':<24}" + "".join(f"{name:>10}" for name in conditioning_sets)
print(header)
for label, path in paths.items():
    row = f"{label:<24}"
    for cond in conditioning_sets.values():
        blocked = is_blocked(path, cond)
        row += f"{'blocked' if blocked else 'OPEN':>10}"
    print(row)

print("\nCompare this table to theory.md section 2.5 -- note the last column: the")
print("collider path flips from blocked to OPEN once C is conditioned on.")
