"""
Clustering Overview — side-by-side comparison of clustering families.

Run:
    pip install numpy matplotlib scikit-learn
    python clustering_overview.py

This script is meant as a map of the territory: it runs K-Means, Agglomerative
(Hierarchical), DBSCAN, Mean-Shift and Spectral Clustering on three synthetic
datasets that stress different assumptions (round blobs, non-convex moons,
anisotropic/stretched blobs) and plots the results together so you can *see*
why each algorithm exists.

Each individual algorithm has its own dedicated, from-scratch implementation
in its own folder (K-Means/code, DBSCAN/code, Hierarchical Clustering/code, ...).
This file only uses scikit-learn, purely to give a fast qualitative comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering


def make_datasets(n_samples=300, random_state=42):
    rng = np.random.RandomState(random_state)

    blobs = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.7, random_state=random_state)

    moons = make_moons(n_samples=n_samples, noise=0.06, random_state=random_state)

    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    transformation = rng.normal(size=(2, 2))
    X_aniso = X @ transformation
    aniso = (X_aniso, y)

    return {"blobs": blobs, "moons": moons, "anisotropic": aniso}


def get_algorithms(n_clusters=3):
    return {
        "KMeans": KMeans(n_clusters=n_clusters, n_init=10, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
        "MeanShift": MeanShift(),
        "Spectral": SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
                                        random_state=42),
    }


def run():
    datasets = make_datasets()
    algo_names = ["KMeans", "Agglomerative", "DBSCAN", "MeanShift", "Spectral"]

    fig, axes = plt.subplots(len(datasets), len(algo_names),
                              figsize=(4 * len(algo_names), 4 * len(datasets)))

    for row, (ds_name, (X, _y)) in enumerate(datasets.items()):
        X = StandardScaler().fit_transform(X)
        algorithms = get_algorithms(n_clusters=3)
        for col, name in enumerate(algo_names):
            ax = axes[row, col]
            model = algorithms[name]
            try:
                labels = model.fit_predict(X)
            except Exception as e:
                labels = np.zeros(X.shape[0])
                print(f"{name} on {ds_name} failed: {e}")
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(ds_name)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = "clustering_overview.png"
    plt.savefig(out_path, dpi=140)
    print(f"Saved comparison figure to {out_path}")


if __name__ == "__main__":
    run()
