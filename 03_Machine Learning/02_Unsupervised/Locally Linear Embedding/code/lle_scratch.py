"""
Locally Linear Embedding from scratch (NumPy only).

Steps: k-NN graph -> local reconstruction weights (sum-to-one constrained
least squares) -> eigen-decomposition of (I-W)^T(I-W) for the embedding.

Run:
    python lle_scratch.py
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class LLEScratch:
    def __init__(self, n_neighbors=10, n_components=2, reg=1e-3):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.embedding_ = None

    def _compute_weights(self, X):
        n, d = X.shape
        k = self.n_neighbors
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, indices = nn.kneighbors(X)

        W = np.zeros((n, n))
        for i in range(n):
            neighbors = indices[i, 1:]  # exclude self
            Z = X[neighbors] - X[i]     # (k, d): neighbor offsets from x_i
            C = Z @ Z.T                 # local covariance / Gram matrix (k, k)

            # regularize for numerical stability (esp. when k > d)
            trace = np.trace(C)
            C += np.eye(k) * self.reg * (trace if trace > 0 else 1.0)

            # solve C w = 1 (Lagrange-multiplier solution to the constrained
            # least-squares problem), then normalize so weights sum to 1
            ones = np.ones(k)
            w = np.linalg.solve(C, ones)
            w /= w.sum()

            W[i, neighbors] = w
        return W

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        W = self._compute_weights(X)

        M = (np.eye(n) - W).T @ (np.eye(n) - W)

        eigenvalues, eigenvectors = np.linalg.eigh(M)  # ascending
        # skip the smallest eigenvalue (~0, trivial constant solution);
        # take the next n_components smallest
        order = np.argsort(eigenvalues)
        selected = order[1:self.n_components + 1]

        self.embedding_ = eigenvectors[:, selected]
        self.eigenvalues_ = eigenvalues[selected]
        return self.embedding_


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll

    X, color = make_swiss_roll(n_samples=400, noise=0.03, random_state=42)

    model = LLEScratch(n_neighbors=8, n_components=2)
    Z = model.fit_transform(X)

    print("Original shape:", X.shape, "-> Embedded shape:", Z.shape)
    corr = np.corrcoef(Z[:, 0], color)[0, 1]
    print(f"Correlation between embedding dim 0 and true manifold coordinate: {corr:.3f}")
