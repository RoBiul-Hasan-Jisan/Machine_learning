"""
NMF with scikit-learn — topic modeling on text, and face-parts decomposition
compared against PCA (illustrates NMF's "parts-based" interpretability).

Run:
    python nmf_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.datasets import fetch_olivetti_faces


def topic_modeling_demo():
    documents = [
        "the cat sat on the mat and the cat purred",
        "dogs bark at cats and dogs love to play fetch",
        "the stock market rose today as tech stocks rallied",
        "investors watched the stock market and tech earnings closely",
        "my cat and my dog are best friends who play together",
        "the central bank raised interest rates affecting the stock market",
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())

    n_topics = 2
    nmf = NMF(n_components=n_topics, init="nndsvd", random_state=0, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_

    print("=== Topic modeling with NMF ===")
    for topic_idx, topic in enumerate(H):
        top_words = feature_names[np.argsort(topic)[::-1][:5]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

    print("\nDocument-topic weights (which topic dominates each doc):")
    for doc_idx, weights in enumerate(W):
        dominant = np.argmax(weights)
        print(f"  doc {doc_idx} -> topic {dominant} (weights={np.round(weights, 2)})")


def face_parts_demo():
    faces = fetch_olivetti_faces(shuffle=True, random_state=0)
    X = faces.data[:200]  # already non-negative (pixel intensities in [0,1])

    n_components = 6
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=300)
    nmf.fit(X)

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X)

    fig, axes = plt.subplots(2, n_components, figsize=(2.2 * n_components, 5))
    for i in range(n_components):
        axes[0, i].imshow(nmf.components_[i].reshape(64, 64), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(pca.components_[i].reshape(64, 64), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("NMF parts ->", loc="left")
    axes[1, 0].set_title("PCA components ->", loc="left")
    plt.tight_layout()
    plt.savefig("nmf_vs_pca_faces.png", dpi=140)
    print("\nSaved NMF vs PCA face components to nmf_vs_pca_faces.png")


if __name__ == "__main__":
    topic_modeling_demo()
    try:
        face_parts_demo()
    except Exception as e:
        print(f"\n(face demo skipped: {e})")
