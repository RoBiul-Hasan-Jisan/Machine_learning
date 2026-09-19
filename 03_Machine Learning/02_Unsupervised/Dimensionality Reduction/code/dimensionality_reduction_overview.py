"""
Dimensionality Reduction Overview — PCA vs Isomap vs LLE vs t-SNE side by side.

Run:
    pip install numpy matplotlib scikit-learn
    python dimensionality_reduction_overview.py

Uses only scikit-learn (this is a "map of the territory" script). Dedicated
from-scratch implementations live in each method's own folder.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE


def embed_all(X, n_components=2, random_state=42):
    return {
        "PCA": PCA(n_components=n_components, random_state=random_state).fit_transform(X),
        "Isomap": Isomap(n_components=n_components, n_neighbors=10).fit_transform(X),
        "LLE": LocallyLinearEmbedding(n_components=n_components, n_neighbors=10,
                                       random_state=random_state).fit_transform(X),
        "t-SNE": TSNE(n_components=n_components, random_state=random_state,
                       init="pca", perplexity=30).fit_transform(X),
    }


def plot_embeddings(embeddings, color, title, out_path):
    fig, axes = plt.subplots(1, len(embeddings), figsize=(4.5 * len(embeddings), 4.5))
    for ax, (name, Z) in zip(axes, embeddings.items()):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap="viridis", s=8)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"Saved {out_path}")


def run():
    # 1) Swiss roll — classic non-linear manifold benchmark
    X_roll, color_roll = make_swiss_roll(n_samples=800, noise=0.05, random_state=42)
    embeddings_roll = embed_all(X_roll)
    plot_embeddings(embeddings_roll, color_roll,
                     "Swiss Roll: PCA (linear) vs manifold learners (non-linear)",
                     "swiss_roll_comparison.png")

    # 2) Digits dataset (64-dim images of handwritten digits 0-9)
    digits = load_digits()
    X_digits, y_digits = digits.data[:500], digits.target[:500]
    embeddings_digits = embed_all(X_digits)
    plot_embeddings(embeddings_digits, y_digits,
                     "Digits dataset (64-D -> 2-D): which method separates digit classes best?",
                     "digits_comparison.png")


if __name__ == "__main__":
    run()
