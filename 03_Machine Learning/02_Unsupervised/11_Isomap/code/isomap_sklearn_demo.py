"""
Isomap with scikit-learn — unrolling a Swiss roll, comparing different
n_neighbors values (too few -> disconnection, too many -> shortcut distortion).

Run:
    python isomap_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA


def main():
    X, color = make_swiss_roll(n_samples=1000, noise=0.05, random_state=42)

    neighbor_settings = [4, 10, 30]
    fig, axes = plt.subplots(1, len(neighbor_settings) + 1, figsize=(5 * (len(neighbor_settings) + 1), 4.5))

    pca_2d = PCA(n_components=2).fit_transform(X)
    axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=color, cmap="viridis", s=8)
    axes[0].set_title("PCA (linear) — roll not unrolled")

    for ax, k in zip(axes[1:], neighbor_settings):
        iso = Isomap(n_neighbors=k, n_components=2)
        Z = iso.fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap="viridis", s=8)
        ax.set_title(f"Isomap, n_neighbors={k}")

    plt.tight_layout()
    plt.savefig("isomap_swiss_roll.png", dpi=140)
    print("Saved comparison to isomap_swiss_roll.png")
    print("Try n_neighbors=4 (may disconnect / noisy) vs 10 (clean) vs 30 "
          "(risk of shortcut distortion across the rolled layers)")


if __name__ == "__main__":
    main()
