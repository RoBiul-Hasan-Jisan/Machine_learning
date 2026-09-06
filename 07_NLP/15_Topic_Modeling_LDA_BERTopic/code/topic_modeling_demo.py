"""
LDA fit on a small synthetic document collection with clear topic
clusters, showing topic-word and document-topic distributions, plus a
simplified embedding-cluster-label pipeline illustrating BERTopic's
core idea.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def make_synthetic_documents(seed=0):
    rng = np.random.default_rng(seed)

    sports_words = ["team", "score", "player", "game", "coach", "match", "goal", "season"]
    finance_words = ["stock", "market", "investor", "economy", "inflation", "profit", "bank", "trade"]
    cooking_words = ["recipe", "flavor", "ingredient", "kitchen", "bake", "chef", "dish", "spice"]

    def make_doc(word_pool, n_words=12):
        return " ".join(rng.choice(word_pool, size=n_words))

    documents, true_labels = [], []
    for label, pool in [("sports", sports_words), ("finance", finance_words), ("cooking", cooking_words)]:
        for _ in range(15):
            documents.append(make_doc(pool))
            true_labels.append(label)

    # a couple of genuinely MIXED documents (two topics blended)
    documents.append(" ".join(list(rng.choice(sports_words, 6)) + list(rng.choice(finance_words, 6))))
    true_labels.append("sports+finance")

    return documents, true_labels


def demo_lda():
    documents, true_labels = make_synthetic_documents()

    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=50)
    doc_topic_matrix = lda.fit_transform(doc_term_matrix)

    print("=== Topic-word distributions (top 6 words per topic) ===")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-6:][::-1]]
        print(f"Topic {topic_idx}: {top_words}")

    print("\n=== Document-topic distributions (first doc of each true category) ===")
    seen_labels = set()
    for i, label in enumerate(true_labels):
        if label not in seen_labels:
            seen_labels.add(label)
            dist = doc_topic_matrix[i].round(3)
            print(f"[{label:15s}] topic mixture: {dist}   doc: '{documents[i][:50]}...'")

    print("\n=== The genuinely MIXED document ===")
    mixed_idx = true_labels.index("sports+finance")
    dist = doc_topic_matrix[mixed_idx].round(3)
    print(f"topic mixture: {dist}")
    print(f"doc: '{documents[mixed_idx]}'")
    print("(A document blending two real topics should get meaningful weight on")
    print("more than one topic, rather than being forced into a single category.)\n")


def simple_word_vectors(vocab_size=50, embedding_dim=8, seed=0):
    """A tiny random 'pretrained' embedding table, standing in for a
    real pretrained embedding model (Lesson 22) in this simplified demo."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(vocab_size, embedding_dim))


def demo_bertopic_style_pipeline():
    """Simplified BERTopic-style pipeline: embed each doc by averaging
    WORD vectors (standing in for a real sentence embedding model),
    cluster with k-means (standing in for HDBSCAN), then extract
    representative words per cluster via class-based TF-IDF."""
    documents, true_labels = make_synthetic_documents()

    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # deterministic "semantic" word vectors: words from the SAME topic pool
    # are given similar vectors, standing in for what a real trained
    # embedding model would discover from co-occurrence patterns (Lesson 03)
    rng = np.random.default_rng(1)
    topic_centers = {
        "sports": rng.normal(0, 1, size=8),
        "finance": rng.normal(0, 1, size=8),
        "cooking": rng.normal(0, 1, size=8),
    }
    sports_words = {"team", "score", "player", "game", "coach", "match", "goal", "season"}
    finance_words = {"stock", "market", "investor", "economy", "inflation", "profit", "bank", "trade"}
    word_vectors = {}
    for w in vocab:
        if w in sports_words:
            center = topic_centers["sports"]
        elif w in finance_words:
            center = topic_centers["finance"]
        else:
            center = topic_centers["cooking"]
        word_vectors[w] = center + rng.normal(0, 0.1, size=8)

    def embed_doc(doc):
        words = doc.split()
        vecs = [word_vectors[w] for w in words if w in word_vectors]
        return np.mean(vecs, axis=0) if vecs else np.zeros(8)

    doc_embeddings = np.array([embed_doc(d) for d in documents])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(doc_embeddings)

    print("=== Simplified BERTopic-style pipeline ===")
    print("Cluster assignment vs true category (first doc of each true category):")
    seen = set()
    for i, label in enumerate(true_labels):
        if label not in seen and label != "sports+finance":
            seen.add(label)
            print(f"  true='{label:10s}' -> assigned cluster {cluster_labels[i]}")

    print("\nRepresentative words per cluster (class-based term frequency):")
    for cluster_id in range(3):
        cluster_docs = [documents[i] for i in range(len(documents)) if cluster_labels[i] == cluster_id]
        combined_text = " ".join(cluster_docs)
        word_counts = {}
        for w in combined_text.split():
            word_counts[w] = word_counts.get(w, 0) + 1
        top_words = sorted(word_counts, key=word_counts.get, reverse=True)[:5]
        print(f"  cluster {cluster_id}: {top_words}")

    print("\nUnlike LDA (word-count based), this pipeline groups documents by")
    print("EMBEDDING similarity -- in a real system with a genuinely trained")
    print("embedding model, this lets documents using different vocabulary for")
    print("the same underlying topic (e.g. 'car' vs 'automobile') land in the")
    print("same cluster, which LDA's bag-of-words approach cannot do.")


if __name__ == "__main__":
    print("=== LDA topic modeling ===")
    demo_lda()

    demo_bertopic_style_pipeline()
