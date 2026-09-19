"""
DBSCAN from scratch (NumPy only).

Implements region queries, core/border/noise point identification, and
cluster expansion via BFS over density-reachable points.

Run:
    python dbscan_scratch.py
"""
import numpy as np
from collections import deque


class DBSCANScratch:
    NOISE = -1

    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.labels_ = None
        self.core_sample_mask_ = None

    def _region_query(self, dist_matrix, idx):
        return np.where(dist_matrix[idx] <= self.eps)[0]

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        diff = X[:, None, :] - X[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        labels = np.full(n, -2, dtype=int)  # -2 = unvisited, -1 = noise, >=0 = cluster id
        is_core = np.zeros(n, dtype=bool)
        cluster_id = -1

        for i in range(n):
            if labels[i] != -2:
                continue  # already processed

            neighbors = self._region_query(dist_matrix, i)
            if len(neighbors) < self.min_pts:
                labels[i] = self.NOISE
                continue

            cluster_id += 1
            is_core[i] = True
            labels[i] = cluster_id

            seeds = deque(neighbors)
            while seeds:
                j = seeds.popleft()
                if labels[j] == self.NOISE:
                    labels[j] = cluster_id  # noise -> border point of this cluster
                if labels[j] != -2:
                    continue  # already assigned to a cluster
                labels[j] = cluster_id
                j_neighbors = self._region_query(dist_matrix, j)
                if len(j_neighbors) >= self.min_pts:
                    is_core[j] = True
                    for k in j_neighbors:
                        if labels[k] in (-2, self.NOISE):
                            seeds.append(k)

        self.labels_ = labels
        self.core_sample_mask_ = is_core
        return self


def k_distance_plot(X, k):
    """Return sorted distances to the k-th nearest neighbor of every point,
    used to pick a good eps (look for the 'knee' of the returned curve)."""
    X = np.asarray(X, dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    dist_matrix.sort(axis=1)
    kth_dists = dist_matrix[:, k]  # column 0 is distance to self (=0)
    return np.sort(kth_dists)


if __name__ == "__main__":
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

    model = DBSCANScratch(eps=0.2, min_pts=5).fit(X)
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise} / {len(X)}")
    print("Cluster sizes:", {c: int(np.sum(labels == c)) for c in sorted(set(labels))})
