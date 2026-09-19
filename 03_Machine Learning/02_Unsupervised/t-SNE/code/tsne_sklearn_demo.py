"""
t-SNE with scikit-learn — digits dataset, perplexity sweep.

Run:
    python tsne_sklearn_demo.py
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


def main():
    digits = load_digits()
    X, y = digits.data, digits.target

    perplexities = [5, 15, 30, 50]
    fig, axes = plt.subplots(1, len(perplexities), figsize=(5 * len(perplexities), 5))

    for ax, perp in zip(axes, perplexities):
        Z = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8)
        ax.set_title(f"perplexity={perp}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("tsne_perplexity_sweep.png", dpi=140)
    print("Saved perplexity sweep to tsne_perplexity_sweep.png")
    print("Notice how low perplexity fragments clusters and high perplexity "
          "smooths/merges them -- neither is 'more correct', they show different scales.")


if __name__ == "__main__":
    main()
