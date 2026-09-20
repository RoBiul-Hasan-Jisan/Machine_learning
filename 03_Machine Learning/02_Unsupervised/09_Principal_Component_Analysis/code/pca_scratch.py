"""
PCA from scratch (NumPy only) — both via covariance eigen-decomposition and
via SVD (shows the two approaches agree up to sign flips).

Run:
    python pca_scratch.py
"""
import numpy as np


class PCAScratch:
    def __init__(self, n_components=2, method="svd"):
        assert method in ("eig", "svd")
        self.n_components = n_components
        self.method = method
        self.mean_ = None
        self.components_ = None          # shape (n_components, n_features)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        n = X.shape[0]

        if self.method == "eig":
            cov = (X_centered.T @ X_centered) / (n - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            self.explained_variance_ = eigenvalues[:self.n_components]
            self.components_ = eigenvectors[:, :self.n_components].T
            total_var = eigenvalues.sum()
        else:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            explained_variance_all = (S ** 2) / (n - 1)
            self.explained_variance_ = explained_variance_all[:self.n_components]
            self.components_ = Vt[:self.n_components]
            total_var = explained_variance_all.sum()

        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X = load_iris().data

    pca_eig = PCAScratch(n_components=2, method="eig").fit(X)
    pca_svd = PCAScratch(n_components=2, method="svd").fit(X)

    print("Explained variance ratio (eig method):", np.round(pca_eig.explained_variance_ratio_, 4))
    print("Explained variance ratio (svd method):", np.round(pca_svd.explained_variance_ratio_, 4))

    Z = pca_svd.transform(X)
    X_reconstructed = pca_svd.inverse_transform(Z)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\nReconstruction MSE using top-2 components: {reconstruction_error:.4f}")
    print("Projected shape:", Z.shape, "(from original", X.shape, ")")
