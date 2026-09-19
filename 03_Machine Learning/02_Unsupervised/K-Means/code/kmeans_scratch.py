"""
K-Means from scratch (NumPy only).

Implements:
  - random initialization
  - k-means++ initialization
  - Lloyd's algorithm (assign / update loop)
  - inertia (WCSS) tracking for the elbow method

Run:
    python kmeans_scratch.py
"""
import numpy as np


class KMeansScratch:
    def __init__(self, n_clusters=3, init="k-means++", max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_random(self, X, rng):
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_kmeanspp(self, X, rng):
        n_samples = X.shape[0]
        centroids = [X[rng.randint(n_samples)]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def _assign(self, X, centroids):
        # (n_samples, n_clusters) distance matrix
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return labels, dists

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        centroids = (self._init_kmeanspp(X, rng) if self.init == "k-means++"
                     else self._init_random(X, rng))

        for iteration in range(self.max_iter):
            labels, dists = self._assign(X, centroids)

            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    # re-seed empty cluster to the farthest point from its centroid
                    farthest = np.argmax(np.min(dists, axis=1))
                    new_centroids[k] = X[farthest]
                else:
                    new_centroids[k] = members.mean(axis=0)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        labels, dists = self._assign(X, centroids)
        inertia = np.sum(np.min(dists, axis=1) ** 2)

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign(X, self.centroids_)
        return labels


def elbow(X, k_range=range(1, 9), init="k-means++", random_state=0):
    """Return list of (k, inertia) to help pick k via the elbow method."""
    results = []
    for k in k_range:
        model = KMeansScratch(n_clusters=k, init=init, random_state=random_state).fit(X)
        results.append((k, model.inertia_))
    return results


if __name__ == "__main__":
    # quick self-test on synthetic blobs
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

    model = KMeansScratch(n_clusters=4, init="k-means++", random_state=42).fit(X)
    print("Converged in", model.n_iter_, "iterations")
    print("Inertia:", round(model.inertia_, 3))
    print("Centroids:\n", np.round(model.centroids_, 3))

    print("\nElbow curve (k, inertia):")
    for k, inertia in elbow(X, range(1, 8)):
        print(f"  k={k}: inertia={inertia:.2f}")
