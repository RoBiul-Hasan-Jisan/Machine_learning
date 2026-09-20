"""
Mean-Shift with scikit-learn — automatic bandwidth estimation, visualized.

Run:
    python meanshift_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth


def main():
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=7)

    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=300, random_state=7)
    print(f"Estimated bandwidth: {bandwidth:.3f}")

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    n_clusters = len(np.unique(ms.labels_))
    print(f"Number of clusters found: {n_clusters}")

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=ms.labels_, cmap="tab10", s=15)
    plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
                c="black", marker="X", s=150, label="modes")
    plt.title(f"Mean-Shift ({n_clusters} clusters found automatically)")
    plt.legend()
    plt.savefig("meanshift_sklearn_demo.png", dpi=140)
    print("Saved plot to meanshift_sklearn_demo.png")


if __name__ == "__main__":
    main()
