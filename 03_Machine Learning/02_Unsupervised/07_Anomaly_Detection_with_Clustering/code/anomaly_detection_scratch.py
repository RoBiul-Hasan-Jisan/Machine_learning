"""
Anomaly Detection with Clustering — from scratch (NumPy only).

Implements:
  - K-Means distance-to-centroid anomaly scoring
  - DBSCAN noise-point anomaly flagging
  - a simplified Local Outlier Factor (LOF)

Run:
    python anomaly_detection_scratch.py
"""
import numpy as np


def kmeans_fit(X, k, n_iter=200, random_state=0):
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
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return centroids, labels, dists


def kmeans_anomaly_scores(X, k=3, random_state=0):
    """Distance from each point to its own cluster's centroid; larger = more anomalous."""
    centroids, labels, dists = kmeans_fit(X, k, random_state=random_state)
    own_dist = dists[np.arange(len(X)), labels]
    return own_dist


def dbscan_noise_flags(X, eps, min_pts):
    """Reuses the DBSCAN idea: returns a boolean mask, True = flagged as noise/anomaly."""
    n = X.shape[0]
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    labels = np.full(n, -2, dtype=int)
    cluster_id = -1
    from collections import deque

    for i in range(n):
        if labels[i] != -2:
            continue
        neighbors = np.where(dist_matrix[i] <= eps)[0]
        if len(neighbors) < min_pts:
            labels[i] = -1
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seeds = deque(neighbors)
        while seeds:
            j = seeds.popleft()
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -2:
                continue
            labels[j] = cluster_id
            j_neighbors = np.where(dist_matrix[j] <= eps)[0]
            if len(j_neighbors) >= min_pts:
                for k_idx in j_neighbors:
                    if labels[k_idx] in (-2, -1):
                        seeds.append(k_idx)

    return labels == -1


def simplified_lof(X, k=10):
    """A simplified Local Outlier Factor. Larger score = more anomalous.

    True LOF uses reachability distance; this simplified version uses the
    ratio of a point's average k-NN distance to the average of its
    neighbors' average k-NN distances (same spirit, easier to read).
    """
    n = X.shape[0]
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    knn_idx = np.argsort(dist_matrix, axis=1)[:, :k]
    local_density = np.array([
        1.0 / (dist_matrix[i, knn_idx[i]].mean() + 1e-12) for i in range(n)
    ])

    lof = np.zeros(n)
    for i in range(n):
        neighbor_densities = local_density[knn_idx[i]]
        lof[i] = neighbor_densities.mean() / (local_density[i] + 1e-12)
    return lof


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    rng = np.random.RandomState(0)
    X_normal, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=0)
    X_outliers = rng.uniform(low=X_normal.min(axis=0) - 3, high=X_normal.max(axis=0) + 3,
                              size=(10, 2))
    X = np.vstack([X_normal, X_outliers])
    y_true_anomaly = np.array([0] * len(X_normal) + [1] * len(X_outliers))

    km_scores = kmeans_anomaly_scores(X, k=3)
    threshold = np.percentile(km_scores, 95)
    km_flag = km_scores > threshold
    print("K-Means distance method — flagged", km_flag.sum(), "points")
    print("  recall on true injected outliers:",
          round(np.mean(km_flag[y_true_anomaly == 1]), 3))

    noise_flag = dbscan_noise_flags(X, eps=1.0, min_pts=5)
    print("\nDBSCAN noise method — flagged", noise_flag.sum(), "points")
    print("  recall on true injected outliers:",
          round(np.mean(noise_flag[y_true_anomaly == 1]), 3))

    lof_scores = simplified_lof(X, k=10)
    lof_threshold = np.percentile(lof_scores, 95)
    lof_flag = lof_scores > lof_threshold
    print("\nSimplified LOF method — flagged", lof_flag.sum(), "points")
    print("  recall on true injected outliers:",
          round(np.mean(lof_flag[y_true_anomaly == 1]), 3))
