"""
K-Means with scikit-learn — elbow method, silhouette analysis, and
MiniBatchKMeans for large data. Use this to validate `kmeans_scratch.py`
and to see the production-grade tooling (k-means++ init, n_init restarts,
silhouette scoring) you'd use in practice.

Run:
    python kmeans_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score


def main():
    X, _ = make_blobs(n_samples=1500, centers=4, cluster_std=0.7, random_state=42)
    X = StandardScaler().fit_transform(X)

    # ---- Elbow method ----
    inertias = []
    k_range = range(1, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)

    # ---- Silhouette analysis (only defined for k >= 2) ----
    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        sil_scores.append(silhouette_score(X, km.labels_))

    best_k = list(range(2, 9))[int(np.argmax(sil_scores))]
    print("Best k by silhouette score:", best_k)

    final_km = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42).fit(X)

    # ---- MiniBatchKMeans comparison (for large-scale data) ----
    mbk = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=64).fit(X)
    print("Full KMeans inertia:", round(final_km.inertia_, 2))
    print("MiniBatchKMeans inertia:", round(mbk.inertia_, 2),
          "(slightly higher is expected — it trades a bit of accuracy for speed)")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(range(2, 9)), sil_scores, marker="o", color="orange")
    axes[1].set_title("Silhouette score vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")

    axes[2].scatter(X[:, 0], X[:, 1], c=final_km.labels_, cmap="tab10", s=12)
    axes[2].scatter(final_km.cluster_centers_[:, 0], final_km.cluster_centers_[:, 1],
                     c="black", marker="X", s=150, label="centroids")
    axes[2].set_title(f"Final clustering (k={best_k})")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("kmeans_sklearn_demo.png", dpi=140)
    print("Saved plots to kmeans_sklearn_demo.png")


if __name__ == "__main__":
    main()
