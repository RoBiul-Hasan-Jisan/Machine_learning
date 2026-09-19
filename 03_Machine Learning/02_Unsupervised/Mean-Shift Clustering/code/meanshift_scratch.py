"""
Mean-Shift Clustering from scratch (NumPy only), using a Gaussian kernel.

Run:
    python meanshift_scratch.py
"""
import numpy as np


class MeanShiftScratch:
    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3, merge_tol=None):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.merge_tol = merge_tol if merge_tol is not None else bandwidth * 0.5
        self.cluster_centers_ = None
        self.labels_ = None

    def _gaussian_mean_shift_step(self, x, X):
        diff = X - x
        sq_dists = np.sum(diff ** 2, axis=1)
        weights = np.exp(-sq_dists / (2 * self.bandwidth ** 2))
        if weights.sum() == 0:
            return x
        return (weights[:, None] * X).sum(axis=0) / weights.sum()

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        shifted = X.copy()

        for i in range(n):
            point = X[i].copy()
            for _ in range(self.max_iter):
                new_point = self._gaussian_mean_shift_step(point, X)
                shift = np.linalg.norm(new_point - point)
                point = new_point
                if shift < self.tol:
                    break
            shifted[i] = point

        # merge converged points that ended up close together into unique modes
        modes = []
        labels = np.full(n, -1, dtype=int)
        for i in range(n):
            assigned = False
            for m_idx, mode in enumerate(modes):
                if np.linalg.norm(shifted[i] - mode) < self.merge_tol:
                    labels[i] = m_idx
                    assigned = True
                    break
            if not assigned:
                modes.append(shifted[i])
                labels[i] = len(modes) - 1

        self.cluster_centers_ = np.array(modes)
        self.labels_ = labels
        return self


def estimate_bandwidth_simple(X, quantile=0.3, random_state=0):
    """Rough bandwidth estimate: quantile of pairwise distances (simplified
    version of sklearn.cluster.estimate_bandwidth)."""
    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    sample_idx = rng.choice(n, size=min(n, 200), replace=False)
    sample = X[sample_idx]
    diff = sample[:, None, :] - sample[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    dists = dists[dists > 0]
    return np.quantile(dists, quantile)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.7, random_state=4)

    bw = estimate_bandwidth_simple(X, quantile=0.2)
    print(f"Estimated bandwidth: {bw:.3f}")

    model = MeanShiftScratch(bandwidth=bw).fit(X)
    print(f"Number of clusters found: {len(model.cluster_centers_)}")
    print("Cluster sizes:", np.bincount(model.labels_))
    print("Cluster centers:\n", np.round(model.cluster_centers_, 2))
