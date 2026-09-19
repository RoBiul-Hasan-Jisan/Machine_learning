"""
PCA with scikit-learn on the digits dataset — scree plot, 2D visualization,
and reconstruction from a reduced number of components.

Run:
    python pca_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def main():
    digits = load_digits()
    X, y = digits.data, digits.target

    # scree plot: how many components needed to explain most of the variance
    pca_full = PCA().fit(X)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_95 = np.argmax(cumulative >= 0.95) + 1
    print(f"Components needed to explain 95% of variance: {n_for_95} (out of {X.shape[1]})")

    # 2D projection for visualization
    pca_2d = PCA(n_components=2).fit_transform(X)

    # reconstruction with a small number of components (compression / denoising)
    pca_k = PCA(n_components=n_for_95).fit(X)
    X_reconstructed = pca_k.inverse_transform(pca_k.transform(X))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].plot(cumulative)
    axes[0, 0].axhline(0.95, color="red", linestyle="--")
    axes[0, 0].set_title("Cumulative explained variance")
    axes[0, 0].set_xlabel("n_components")

    sc = axes[0, 1].scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, cmap="tab10", s=8)
    axes[0, 1].set_title("Digits projected to 2D")
    plt.colorbar(sc, ax=axes[0, 1])

    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    # show original vs reconstructed digits
    for i in range(4):
        axes[1, i].imshow(X[i].reshape(8, 8), cmap="gray")
        axes[1, i].set_title(f"original (label {y[i]})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("pca_sklearn_demo.png", dpi=140)
    print("Saved plots to pca_sklearn_demo.png")

    # separate figure: original vs reconstructed comparison
    fig2, axes2 = plt.subplots(2, 6, figsize=(12, 4.5))
    for i in range(6):
        axes2[0, i].imshow(X[i].reshape(8, 8), cmap="gray")
        axes2[0, i].axis("off")
        axes2[1, i].imshow(X_reconstructed[i].reshape(8, 8), cmap="gray")
        axes2[1, i].axis("off")
    axes2[0, 0].set_ylabel("original")
    axes2[1, 0].set_ylabel("reconstructed")
    fig2.suptitle(f"Original vs reconstruction using {n_for_95} components")
    plt.tight_layout()
    plt.savefig("pca_reconstruction.png", dpi=140)
    print("Saved reconstruction comparison to pca_reconstruction.png")


if __name__ == "__main__":
    main()
