"""
Spectral Clustering from scratch (NumPy only).

Steps: build an RBF similarity graph -> normalized graph Laplacian ->
top-k eigenvectors -> KMeans on the embedding.

Run:
    python spectral_scratch.py
"""
import numpy as np


def rbf_similarity(X, sigma=1.0):
    diff = X[:, None, :] - X[None, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    return np.exp(-sq_dists / (2 * sigma ** 2))


def simple_kmeans(X, k, n_iter=100, random_state=0):
    rng = np.random.RandomState(random_state)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels


class SpectralClusteringScratch:
    def __init__(self, n_clusters=2, sigma=1.0, random_state=0):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.random_state = random_state
        self.labels_ = None
        self.eigenvalues_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        W = rbf_similarity(X, sigma=self.sigma)
        np.fill_diagonal(W, 0.0)  # no self-loops

        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(np.diag(D), 1e-12, None)))
        L_sym = np.eye(len(X)) - D_inv_sqrt @ W @ D_inv_sqrt  # normalized Laplacian

        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)  # ascending order
        self.eigenvalues_ = eigenvalues

        U = eigenvectors[:, :self.n_clusters]
        # row-normalize (standard practice, improves K-Means step)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1
        U_norm = U / norms

        self.labels_ = simple_kmeans(U_norm, self.n_clusters, random_state=self.random_state)
        return self


if __name__ == "__main__":
    from sklearn.datasets import make_circles

    X, y_true = make_circles(n_samples=200, factor=0.4, noise=0.05, random_state=0)

    model = SpectralClusteringScratch(n_clusters=2, sigma=0.15).fit(X)
    print("Cluster sizes:", np.bincount(model.labels_))
    # Compare with the true concentric-circle labels via a simple accuracy
    # check (allowing for label-permutation, since cluster ids are arbitrary)
    acc1 = np.mean(model.labels_ == y_true)
    acc2 = np.mean(model.labels_ == (1 - y_true))
    print("Agreement with true rings:", round(max(acc1, acc2), 3))
    print("First 10 eigenvalues of the Laplacian:", np.round(model.eigenvalues_[:10], 4))
