"""

Combine many features into fewer, denser ones (vs. Section 07, which
drops features outright).

PCA and SVD below are exact, standard implementations. t-SNE and UMAP
are genuinely complex algorithms (adaptive perplexity search, graph
construction, specialized optimizers) -- full from-scratch versions
would be hundreds of lines and still not match production
implementations. What's included here are simplified-but-real versions
that capture the core mechanism of each (t-SNE: gradient descent on a
KL divergence between neighbor-probability distributions; UMAP: a
force-directed layout on a fuzzy neighbor graph) so you can see how
they actually work, clearly labeled as simplified. For production use,
use sklearn.manifold.TSNE and the umap-learn package.
"""

import numpy as np



# PCA


class PCA:
    """
    Finds orthogonal directions of maximum variance and projects data
    onto the top `n_components` of them. Implemented via eigendecomposition
    of the covariance matrix (the classic textbook approach; see the SVD
    class below for the numerically preferred alternative on the same data).

    IMPORTANT: PCA is scale-sensitive -- a feature with a much larger raw
    range will dominate the top components purely due to units, not real
    importance. Scale your features first (05_Feature_Scaling) unless
    you specifically want raw-variance-driven components.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh: cov is symmetric

        # eigh returns ascending order; we want descending (largest variance first)
        order = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[:, :self.n_components]
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)



# SVD


def truncated_svd(X, n_components=2):
    """
    Singular Value Decomposition, truncated to the top n_components.
    X = U @ diag(S) @ Vt. The top-k left singular vectors scaled by
    their singular values (U[:, :k] * S[:k]) give the same reduced
    representation PCA would on centered data -- but SVD works directly
    on X without requiring centering first, which matters for sparse
    data (e.g. a TF-IDF matrix) where centering would destroy sparsity.

    Returns:
        X_reduced: (n_samples, n_components) projected data
        explained_variance_ratio: fraction of total squared-singular-value
            "energy" captured by each kept component
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_reduced = U[:, :n_components] * S[:n_components]
    explained_variance_ratio = (S[:n_components] ** 2) / (S ** 2).sum()
    return X_reduced, explained_variance_ratio



# t-SNE (simplified)


def _pairwise_sq_dists(X):
    sum_sq = (X ** 2).sum(axis=1)
    return sum_sq[:, None] + sum_sq[None, :] - 2 * X @ X.T


def _compute_p_matrix(X, perplexity, tol=1e-5, max_iter=50):
    """
    Convert distances into a symmetric joint probability matrix P, where
    P[i,j] is high if point j is a "close neighbor" of point i. Each
    row's effective neighborhood size is tuned (via binary search on a
    per-point precision/sigma) so it matches the target `perplexity` --
    this is the "adaptive bandwidth" that makes t-SNE handle regions of
    varying density better than a fixed-radius neighbor definition.
    """
    n = X.shape[0]
    dist_sq = _pairwise_sq_dists(X)
    P = np.zeros((n, n))
    target_entropy = np.log(perplexity)

    for i in range(n):
        lo, hi = -np.inf, np.inf
        beta = 1.0  # beta = 1 / (2 * sigma_i^2)

        for _ in range(max_iter):
            row = -dist_sq[i].copy()
            row[i] = -np.inf
            row_shifted = row - np.max(row[row > -np.inf])
            p_row = np.exp(beta * row_shifted)
            p_row[i] = 0.0
            sum_p = p_row.sum()
            if sum_p == 0:
                sum_p = 1e-12
            p_row /= sum_p

            entropy = -np.sum(p_row[p_row > 0] * np.log(p_row[p_row > 0]))
            diff = entropy - target_entropy

            if abs(diff) < tol:
                break
            if diff > 0:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2

        P[i] = p_row

    # symmetrize and normalize (standard t-SNE joint-probability construction)
    P = (P + P.T) / (2 * n)
    return np.maximum(P, 1e-12)


def simple_tsne(X, n_components=2, perplexity=15.0, n_iter=300,
                 learning_rate=100.0, random_state=42):
    """
    Simplified t-SNE: minimizes KL divergence between a high-dimensional
    neighbor-probability distribution P (see _compute_p_matrix) and a
    low-dimensional Student-t-based neighbor distribution Q, via plain
    gradient descent (no momentum/early-exaggeration schedule, which
    production implementations use for better convergence).

    VISUALIZATION ONLY: distances between well-separated clusters in the
    output are not meaningful (only local neighborhoods are preserved),
    and this is not meant to feed into a downstream model.
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]

    P = _compute_p_matrix(X, perplexity)
    Y = rng.normal(0, 1e-4, size=(n, n_components))

    for _ in range(n_iter):
        sum_y_sq = (Y ** 2).sum(axis=1)
        num = 1.0 / (1.0 + sum_y_sq[:, None] + sum_y_sq[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)

        pq_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff_y = Y[i] - Y  # (n, n_components)
            grad[i] = 4 * np.sum((pq_diff[i] * num[i])[:, None] * diff_y, axis=0)

        Y -= learning_rate * grad

    return Y



# UMAP (simplified)


def simple_umap(X, n_components=2, n_neighbors=10, n_iter=200,
                 learning_rate=1.0, random_state=42):
    """
    Simplified UMAP: builds a k-nearest-neighbor graph (fuzzy edge weights
    based on relative distance to each point's neighbors), then optimizes
    a low-dimensional layout so that connected points attract and
    unconnected points repel -- a force-directed graph layout, which is
    the conceptual core of UMAP. Production UMAP uses a more careful
    fuzzy-set-union graph construction and a stochastic negative-sampling
    optimizer (much faster on large data); this version favors
    conceptual clarity over speed/accuracy.

    VISUALIZATION (or lightweight preprocessing) ONLY.
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]

    dist_sq = _pairwise_sq_dists(X)
    np.fill_diagonal(dist_sq, np.inf)
    neighbor_idx = np.argsort(dist_sq, axis=1)[:, :n_neighbors]

    # build a symmetric weighted adjacency: weight decays with distance rank
    weights = np.zeros((n, n))
    for i in range(n):
        for rank, j in enumerate(neighbor_idx[i]):
            weights[i, j] = max(weights[i, j], 1.0 / (1.0 + rank))
    weights = np.maximum(weights, weights.T)  # symmetrize (fuzzy union, simplified)

    Y = rng.normal(0, 1.0, size=(n, n_components))

    for it in range(n_iter):
        # attraction along graph edges
        attr_grad = np.zeros_like(Y)
        for i in range(n):
            for j in range(n):
                if weights[i, j] > 0:
                    diff = Y[i] - Y[j]
                    dist_sq_ij = (diff ** 2).sum() + 1e-8
                    attr_grad[i] += -weights[i, j] * diff / (1 + dist_sq_ij)

        # repulsion from a random sample of non-neighbors (negative sampling, simplified)
        neg_idx = rng.randint(0, n, size=(n, 5))
        rep_grad = np.zeros_like(Y)
        for i in range(n):
            for j in neg_idx[i]:
                if j == i or weights[i, j] > 0:
                    continue
                diff = Y[i] - Y[j]
                dist_sq_ij = (diff ** 2).sum() + 1e-8
                rep_grad[i] += diff / (dist_sq_ij * (1 + dist_sq_ij))

        Y += learning_rate * (attr_grad + 0.1 * rep_grad)

    return Y






def _demo():
    
    print("  DIMENSIONALITY REDUCTION DEMO")
    

    rng = np.random.RandomState(0)
    # three separated blobs in 5D
    n_per = 25
    blob1 = rng.normal(0, 0.5, (n_per, 5))
    blob2 = rng.normal(5, 0.5, (n_per, 5))
    blob3 = rng.normal([0, 5, 0, 5, 0], 0.5, (n_per, 5))
    X = np.vstack([blob1, blob2, blob3])
    labels = np.array([0] * n_per + [1] * n_per + [2] * n_per)

    print(f"\nData: {X.shape[0]} points in {X.shape[1]}D, 3 known blobs")

    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    print(f"\nPCA -> 2D. Explained variance ratio: {np.round(pca.explained_variance_ratio_, 3)}")
    print("Per-blob mean position (should be well-separated if PCA captured cluster structure):")
    for c in [0, 1, 2]:
        print(f"  blob {c}: mean={np.round(X_pca[labels == c].mean(axis=0), 2)}")

    X_svd, var_ratio = truncated_svd(X - X.mean(axis=0), n_components=2)
    print(f"\nSVD -> 2D. Explained variance ratio: {np.round(var_ratio, 3)}")
    print("(on centered data, SVD's projection matches PCA's up to sign)")

    print("\nRunning simplified t-SNE (this does real gradient descent, ~few seconds)...")
    X_tsne = simple_tsne(X, n_components=2, perplexity=10, n_iter=200, random_state=0)
    print("Per-blob mean position in t-SNE space:")
    for c in [0, 1, 2]:
        print(f"  blob {c}: mean={np.round(X_tsne[labels == c].mean(axis=0), 2)}")

    print("\nRunning simplified UMAP...")
    X_umap = simple_umap(X, n_components=2, n_neighbors=8, n_iter=100, random_state=0)
    print("Per-blob mean position in UMAP space:")
    for c in [0, 1, 2]:
        print(f"  blob {c}: mean={np.round(X_umap[labels == c].mean(axis=0), 2)}")

    print("\nNote: for both t-SNE and UMAP, what matters is that the three")
    print("blobs land in well-SEPARATED regions -- not the absolute coordinates")
    print("or distances between blob centers, which aren't meaningful here.")


if __name__ == "__main__":
    _demo()
