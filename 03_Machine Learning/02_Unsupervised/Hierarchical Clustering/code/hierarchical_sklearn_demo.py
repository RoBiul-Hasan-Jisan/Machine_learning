"""
Hierarchical Clustering with scipy / scikit-learn — real dendrogram plotting
and a flat clustering via AgglomerativeClustering. Use this to visualize what
`hierarchical_scratch.py` computes numerically, and to compare linkage rules.

Run:
    python hierarchical_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering


def main():
    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.6, random_state=1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    linkages = ["single", "complete", "average", "ward"]

    for ax, method in zip(axes, linkages):
        Z = linkage(X, method=method)
        dendrogram(Z, ax=ax, no_labels=True)
        ax.set_title(f"{method} linkage dendrogram")

    plt.tight_layout()
    plt.savefig("dendrograms.png", dpi=140)
    print("Saved dendrograms to dendrograms.png")

    # Flat clustering via Ward linkage, cut for k=3
    Z = linkage(X, method="ward")
    flat_labels_scipy = fcluster(Z, t=3, criterion="maxclust")

    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    flat_labels_sklearn = model.fit_predict(X)

    print("scipy fcluster label counts:", np.bincount(flat_labels_scipy))
    print("sklearn AgglomerativeClustering label counts:", np.bincount(flat_labels_sklearn))

    # cophenetic correlation: how well does the dendrogram preserve
    # the *actual* pairwise distances? (closer to 1 is better)
    from scipy.cluster.hierarchy import cophenet
    coph_dists = cophenet(Z)
    orig_dists = pdist(X)
    corr = np.corrcoef(coph_dists, orig_dists)[0, 1]
    print(f"Cophenetic correlation (ward): {corr:.3f}")


if __name__ == "__main__":
    main()
