"""
09. Causal Graphs 

A minimal, dependency-free DAG toolkit:
  - parents / children / ancestors / descendants of a node
  - enumerate ALL paths (not just directed paths) between two nodes
  - classify a 3-node subpath (through a middle node) as chain / fork / collider

Applied to Module 03's confounder DAG:  Z -> T,  Z -> Y,  T -> Y

Run with: python dag_structure_demo.py
"""

from itertools import combinations

# Represent the DAG as a set of directed edges (parent, child).
EDGES = {
    ("Z", "T"),
    ("Z", "Y"),
    ("T", "Y"),
}
NODES = {n for edge in EDGES for n in edge}


def parents(node):
    return {a for (a, b) in EDGES if b == node}


def children(node):
    return {b for (a, b) in EDGES if a == node}


def ancestors(node):
    result = set()
    frontier = parents(node)
    while frontier:
        result |= frontier
        frontier = {p for n in frontier for p in parents(n)} - result
    return result


def descendants(node):
    result = set()
    frontier = children(node)
    while frontier:
        result |= frontier
        frontier = {c for n in frontier for c in children(n)} - result
    return result


def undirected_adjacency():
    adj = {n: set() for n in NODES}
    for a, b in EDGES:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def all_paths(start, end):
    """Enumerate all simple paths between start and end, ignoring edge direction."""
    adj = undirected_adjacency()
    paths = []

    def dfs(node, visited, path):
        if node == end:
            paths.append(list(path))
            return
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor, visited | {neighbor}, path + [neighbor])

    dfs(start, {start}, [start])
    return paths


def classify_junction(a, m, b):
    """Classify the 3-node subpath a -- m -- b as chain, fork, or collider."""
    a_to_m = (a, m) in EDGES
    m_to_a = (m, a) in EDGES
    m_to_b = (m, b) in EDGES
    b_to_m = (b, m) in EDGES

    if a_to_m and m_to_b:
        return "chain (a -> m -> b)"
    if m_to_a and m_to_b:
        return "fork (a <- m -> b)"
    if a_to_m and b_to_m:
        return "collider (a -> m <- b)"
    return "mixed/other"


print("=== Graph structure: parents, children, ancestors, descendants ===")
for node in sorted(NODES):
    print(f"{node}: parents={parents(node)}, children={children(node)}, "
          f"ancestors={ancestors(node)}, descendants={descendants(node)}")

print("\n=== All paths between T and Y (ignoring direction) ===")
for path in all_paths("T", "Y"):
    # Describe whether each edge along the path follows or goes against arrow direction
    directions = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        directions.append(f"{a}->{b}" if (a, b) in EDGES else f"{a}<-{b}")
    is_causal = all((path[i], path[i + 1]) in EDGES for i in range(len(path) - 1))
    print(f"  {' - '.join(directions)}   {'(CAUSAL path)' if is_causal else '(NON-CAUSAL path)'}")

print("\n=== Junction classification for every 3-node subpath ===")
for a, m, b in [("T", "Z", "Y")]:
    print(f"  {a} -- {m} -- {b}: {classify_junction(a, m, b)}")
