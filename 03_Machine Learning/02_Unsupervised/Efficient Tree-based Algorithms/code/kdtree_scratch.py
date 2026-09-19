"""
KD-Tree from scratch (pure Python) — recursive construction plus a
branch-and-bound nearest-neighbor query, validated against brute force.

Run:
    python kdtree_scratch.py
"""
import math


class KDNode:
    __slots__ = ("point", "index", "axis", "left", "right")

    def __init__(self, point, index, axis, left=None, right=None):
        self.point = point
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, points):
        self.k = len(points[0])
        indexed_points = list(enumerate(points))
        self.root = self._build(indexed_points, depth=0)

    def _build(self, indexed_points, depth):
        if not indexed_points:
            return None
        axis = depth % self.k
        indexed_points.sort(key=lambda ip: ip[1][axis])
        mid = len(indexed_points) // 2
        idx, point = indexed_points[mid]
        return KDNode(
            point=point,
            index=idx,
            axis=axis,
            left=self._build(indexed_points[:mid], depth + 1),
            right=self._build(indexed_points[mid + 1:], depth + 1),
        )

    @staticmethod
    def _sq_dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def nearest_neighbor(self, query):
        """Branch-and-bound nearest-neighbor search. Returns (index, point, sq_dist)."""
        best = [None]  # mutable holder so the recursive closure can update it

        def search(node):
            if node is None:
                return
            d = self._sq_dist(query, node.point)
            if best[0] is None or d < best[0][2]:
                best[0] = (node.index, node.point, d)

            axis = node.axis
            diff = query[axis] - node.point[axis]
            close_branch, far_branch = (node.left, node.right) if diff < 0 else (node.right, node.left)

            search(close_branch)

            # only explore the far branch if it could possibly contain a
            # closer point than our current best -- this pruning is the
            # entire point of using a tree instead of brute force
            if best[0] is None or diff ** 2 < best[0][2]:
                search(far_branch)

        search(self.root)
        return best[0]


def brute_force_nearest(points, query):
    best_idx, best_dist = None, math.inf
    for i, p in enumerate(points):
        d = KDTree._sq_dist(query, p)
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx, points[best_idx], best_dist


if __name__ == "__main__":
    import random

    rng = random.Random(0)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(500)]

    tree = KDTree(points)

    queries = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(20)]
    all_match = True
    for q in queries:
        kd_idx, kd_point, kd_dist = tree.nearest_neighbor(q)
        bf_idx, bf_point, bf_dist = brute_force_nearest(points, q)
        match = abs(kd_dist - bf_dist) < 1e-9
        all_match &= match
        if not match:
            print(f"MISMATCH for query {q}: kdtree={kd_point} (d={kd_dist:.3f}), "
                  f"brute={bf_point} (d={bf_dist:.3f})")

    print(f"Checked {len(queries)} queries against {len(points)} points.")
    print("All KD-tree results match brute force:", all_match)
