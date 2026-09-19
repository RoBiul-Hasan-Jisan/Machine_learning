"""
Spectral Clustering with scikit-learn — concentric circles (non-convex),
compared against KMeans, plus an eigenvalue-gap plot to help pick k.

Run:
    python spectral_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix


def main():
    X, _ = make_circles(n_samples=400, factor=0.4, noise=0.05, random_state=0)

    spectral = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                                   n_neighbors=10, random_state=0).fit(X)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    # eigenvalue gap plot (helps choose k): build a k-NN graph, compute the
    # normalized Laplacian, and look at its smallest eigenvalues.
    graph = kneighbors_graph(X, n_neighbors=10, include_self=False)
    graph = 0.5 * (graph + graph.T)  # symmetrize
    L = laplacian(csr_matrix(graph), normed=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(L.toarray()))[:10]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="tab10", s=15)
    axes[0].set_title("KMeans — fails on concentric circles")

    axes[1].scatter(X[:, 0], X[:, 1], c=spectral.labels_, cmap="tab10", s=15)
    axes[1].set_title("Spectral Clustering — correct")

    axes[2].plot(range(1, 11), eigenvalues, marker="o")
    axes[2].set_title("Smallest 10 Laplacian eigenvalues")
    axes[2].set_xlabel("index")
    axes[2].set_ylabel("eigenvalue")

    plt.tight_layout()
    plt.savefig("spectral_vs_kmeans.png", dpi=140)
    print("Saved comparison to spectral_vs_kmeans.png")


if __name__ == "__main__":
    main()
