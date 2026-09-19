"""
DBSCAN with scikit-learn — k-distance plot for choosing eps, and a head-to-head
comparison against KMeans on the two-moons dataset (arbitrary-shape clusters).

Run:
    python dbscan_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def main():
    X, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
    X = StandardScaler().fit_transform(X)

    min_pts = 5
    nn = NearestNeighbors(n_neighbors=min_pts).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    dbscan = DBSCAN(eps=0.2, min_samples=min_pts).fit(X)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    n_clusters_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    print(f"DBSCAN found {n_clusters_db} clusters, "
          f"{np.sum(dbscan.labels_ == -1)} noise points")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(k_distances)
    axes[0].set_title(f"k-distance plot (k={min_pts})")
    axes[0].set_xlabel("points sorted by distance")
    axes[0].set_ylabel(f"distance to {min_pts}-th neighbor")
    axes[0].axhline(0.2, color="red", linestyle="--", label="chosen eps=0.2")
    axes[0].legend()

    axes[1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="tab10", s=15)
    axes[1].set_title("KMeans (k=2) — fails on non-convex shape")

    axes[2].scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap="tab10", s=15)
    axes[2].set_title("DBSCAN — correctly separates the moons")

    plt.tight_layout()
    plt.savefig("dbscan_vs_kmeans.png", dpi=140)
    print("Saved comparison to dbscan_vs_kmeans.png")


if __name__ == "__main__":
    main()
