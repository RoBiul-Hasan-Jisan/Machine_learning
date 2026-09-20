"""
t-SNE from scratch (NumPy only) — naive O(n^2) version for learning purposes.

Implements:
  - per-point sigma search via binary search to hit a target perplexity
  - symmetrized high-dimensional affinities (P)
  - Student-t low-dimensional affinities (Q)
  - gradient descent on the KL divergence between P and Q

Run:
    python tsne_scratch.py
"""
import numpy as np


def _pairwise_sq_dists(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sum(diff ** 2, axis=2)


def _perplexity_from_sigma(sq_dists_row, sigma, i):
    """Given squared distances from point i to all others and a sigma,
    compute the resulting Gaussian conditional distribution's perplexity."""
    exponent = -sq_dists_row / (2 * sigma ** 2)
    exponent -= exponent.max()  # numerical stability
    p = np.exp(exponent)
    p[i] = 0.0
    p_sum = p.sum()
    if p_sum == 0:
        return 0.0, p
    p /= p_sum
    entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    perplexity = 2 ** entropy
    return perplexity, p


def _find_sigma_for_perplexity(sq_dists_row, i, target_perplexity, tol=1e-5, max_iter=50):
    sigma_min, sigma_max = 1e-4, 1000.0
    for _ in range(max_iter):
        sigma = (sigma_min + sigma_max) / 2
        perplexity, p = _perplexity_from_sigma(sq_dists_row, sigma, i)
        if abs(perplexity - target_perplexity) < tol:
            break
        if perplexity > target_perplexity:
            sigma_max = sigma
        else:
            sigma_min = sigma
    return p


def compute_high_dim_affinities(X, perplexity=20):
    n = X.shape[0]
    sq_dists = _pairwise_sq_dists(X)
    P_cond = np.zeros((n, n))
    for i in range(n):
        P_cond[i] = _find_sigma_for_perplexity(sq_dists[i], i, perplexity)
    P = (P_cond + P_cond.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    return P


def compute_low_dim_affinities(Y):
    sq_dists = _pairwise_sq_dists(Y)
    inv = 1.0 / (1.0 + sq_dists)
    np.fill_diagonal(inv, 0.0)
    Q = inv / inv.sum()
    Q = np.maximum(Q, 1e-12)
    return Q, inv


def tsne_scratch(X, n_components=2, perplexity=20, n_iter=500, lr=200.0,
                  random_state=0, verbose_every=100):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    rng = np.random.RandomState(random_state)

    P = compute_high_dim_affinities(X, perplexity=perplexity)
    Y = rng.normal(scale=1e-4, size=(n, n_components))

    Y_prev1 = Y.copy()
    Y_prev2 = Y.copy()
    momentum = 0.5

    for it in range(n_iter):
        Q, inv = compute_low_dim_affinities(Y)
        PQ_diff = P - Q  # (n, n)

        grad = np.zeros_like(Y)
        for i in range(n):
            factor = (PQ_diff[i] * inv[i])[:, None]  # (n, 1)
            grad[i] = 4 * np.sum(factor * (Y[i] - Y), axis=0)

        if it > 100:
            momentum = 0.8

        Y_new = Y - lr * grad + momentum * (Y_prev1 - Y_prev2)
        Y_prev2 = Y_prev1
        Y_prev1 = Y
        Y = Y_new

        if verbose_every and it % verbose_every == 0:
            kl = np.sum(P * np.log(P / Q))
            print(f"iter {it:4d}  KL divergence = {kl:.4f}")

    return Y


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    digits = load_digits()
    X, y = digits.data[:150], digits.target[:150]  # small subset: O(n^2) is slow

    Y = tsne_scratch(X, perplexity=15, n_iter=400, lr=100.0, random_state=0)

    # quick sanity check: same-digit points should be closer than
    # different-digit points on average, in the resulting embedding
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(Y))
    same_label = y[:, None] == y[None, :]
    np.fill_diagonal(same_label, False)
    avg_same = dist_matrix[same_label].mean()
    avg_diff = dist_matrix[~same_label & ~np.eye(len(y), dtype=bool)].mean()
    print(f"\nAvg embedding distance, same digit: {avg_same:.2f}")
    print(f"Avg embedding distance, different digit: {avg_diff:.2f}")
    print("(same-digit distance should be noticeably smaller if t-SNE worked)")
