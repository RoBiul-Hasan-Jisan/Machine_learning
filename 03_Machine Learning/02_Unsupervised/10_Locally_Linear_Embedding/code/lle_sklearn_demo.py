"""
LLE with scikit-learn — Swiss roll and S-curve, standard vs modified LLE.

Run:
    python lle_sklearn_demo.py
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.manifold import LocallyLinearEmbedding


def main():
    datasets = {
        "Swiss roll": make_swiss_roll(n_samples=800, noise=0.05, random_state=0),
        "S-curve": make_s_curve(n_samples=800, noise=0.05, random_state=0),
    }
    methods = ["standard", "modified", "hessian", "ltsa"]

    fig, axes = plt.subplots(len(datasets), len(methods),
                              figsize=(4.5 * len(methods), 4.5 * len(datasets)))

    for row, (name, (X, color)) in enumerate(datasets.items()):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            n_neighbors = 12 if method != "standard" else 10
            lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2,
                                          method=method, random_state=0)
            try:
                Z = lle.fit_transform(X)
                ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap="viridis", s=8)
            except Exception as e:
                ax.text(0.5, 0.5, f"failed:\n{e}", ha="center", va="center", fontsize=8)
            if row == 0:
                ax.set_title(method)
            if col == 0:
                ax.set_ylabel(name)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("lle_methods_comparison.png", dpi=140)
    print("Saved comparison to lle_methods_comparison.png")


if __name__ == "__main__":
    main()
