"""
Non-negative Matrix Factorization from scratch (NumPy only), using the
Lee & Seung multiplicative update rules for the squared-error objective.

Run:
    python nmf_scratch.py
"""
import numpy as np


class NMFScratch:
    def __init__(self, n_components=2, n_iter=300, eps=1e-10, random_state=0):
        self.n_components = n_components
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        self.W_ = None
        self.H_ = None
        self.errors_ = None

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        assert np.all(X >= 0), "NMF requires a non-negative input matrix"
        n, d = X.shape
        k = self.n_components
        rng = np.random.RandomState(self.random_state)

        W = rng.uniform(0, 1, size=(n, k))
        H = rng.uniform(0, 1, size=(k, d))

        errors = []
        for it in range(self.n_iter):
            # update H
            numerator_H = W.T @ X
            denominator_H = W.T @ W @ H + self.eps
            H *= numerator_H / denominator_H

            # update W
            numerator_W = X @ H.T
            denominator_W = W @ H @ H.T + self.eps
            W *= numerator_W / denominator_W

            error = np.linalg.norm(X - W @ H, ord="fro")
            errors.append(error)

        self.W_ = W
        self.H_ = H
        self.errors_ = errors
        return W

    def inverse_transform(self, W):
        return W @ self.H_


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # synthetic non-negative data: 3 "true" additive parts mixed together
    true_H = rng.uniform(0, 1, size=(3, 10))
    true_W = rng.uniform(0, 1, size=(200, 3))
    X = true_W @ true_H  # noiseless non-negative data with rank 3

    model = NMFScratch(n_components=3, n_iter=500, random_state=1)
    W = model.fit_transform(X)
    X_reconstructed = model.inverse_transform(W)

    print("Reconstruction error (start):", round(model.errors_[0], 4))
    print("Reconstruction error (end):  ", round(model.errors_[-1], 4))
    print("All entries of W, H non-negative:", np.all(W >= 0), np.all(model.H_ >= 0))
    print("Relative reconstruction MSE:",
          round(float(np.mean((X - X_reconstructed) ** 2) / np.mean(X ** 2)), 6))
