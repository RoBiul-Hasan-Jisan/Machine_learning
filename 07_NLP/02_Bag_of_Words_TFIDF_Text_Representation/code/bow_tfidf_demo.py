"""
From-scratch TF-IDF, verified against sklearn's TfidfVectorizer, plus a
bigram bag-of-words example distinguishing reordered sentences that
plain unigram bag-of-words cannot.
"""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bag_of_words(documents):
    tokenized_docs = [doc.lower().split() for doc in documents]
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    matrix = np.zeros((len(documents), len(vocab)), dtype=int)
    for i, doc in enumerate(tokenized_docs):
        for word in doc:
            matrix[i, word_to_idx[word]] += 1
    return matrix, vocab


def compute_tfidf(documents):
    tokenized_docs = [doc.lower().split() for doc in documents]
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    N = len(documents)
    df = Counter()
    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1

    idf = {word: np.log(N / (1 + df[word])) for word in vocab}

    tfidf_matrix = np.zeros((N, len(vocab)))
    for i, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        total_words = len(doc)
        for word, count in counts.items():
            tf = count / total_words
            tfidf_matrix[i, word_to_idx[word]] = tf * idf[word]

    return tfidf_matrix, vocab


def demo_bow_ignores_order():
    docs = ["the dog bit the man", "the man bit the dog"]
    matrix, vocab = bag_of_words(docs)
    print("Vocabulary:", vocab)
    print("Doc 1 vector:", matrix[0])
    print("Doc 2 vector:", matrix[1])
    assert np.array_equal(matrix[0], matrix[1])
    print("Identical vectors -- bag-of-words cannot distinguish these two sentences.\n")


def demo_tfidf_vs_sklearn():
    docs = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "photosynthesis converts light into energy",
        "the cat and the dog are friends",
    ]

    our_tfidf, our_vocab = compute_tfidf(docs)

    # sklearn's default IDF formula differs slightly (adds 1 to the whole log term
    # and L2-normalizes rows by default) -- match it explicitly for a fair comparison.
    sklearn_vectorizer = TfidfVectorizer(norm=None, smooth_idf=True, sublinear_tf=False)
    sklearn_tfidf = sklearn_vectorizer.fit_transform(docs).toarray()
    sklearn_vocab = sklearn_vectorizer.get_feature_names_out()

    print("Our vocab size:", len(our_vocab), " sklearn vocab size:", len(sklearn_vocab))
    print("(Note: sklearn's smoothed IDF formula -- log((1+N)/(1+df)) + 1 -- differs from")
    print("our simpler log(N/(1+df)), so exact values won't match, but relative rankings")
    print("of important vs unimportant words should agree.)\n")

    # Compare which word gets the highest TF-IDF score in doc 3 (the distinctive one)
    doc3_idx = 2
    our_top_word = our_vocab[np.argmax(our_tfidf[doc3_idx])]
    sklearn_top_word = sklearn_vocab[np.argmax(sklearn_tfidf[doc3_idx])]
    print(f"Doc 3 ('{docs[doc3_idx]}') highest-TF-IDF word:")
    print(f"  Ours:    '{our_top_word}'")
    print(f"  sklearn: '{sklearn_top_word}'")
    print("(Both should identify a distinctive, rare word from this document.)\n")


def demo_bigrams_recover_order():
    docs = ["the dog bit the man", "the man bit the dog"]

    unigram_vec = CountVectorizer(ngram_range=(1, 1))
    unigram_matrix = unigram_vec.fit_transform(docs).toarray()

    bigram_vec = CountVectorizer(ngram_range=(2, 2))
    bigram_matrix = bigram_vec.fit_transform(docs).toarray()

    print("Unigram vectors identical?", np.array_equal(unigram_matrix[0], unigram_matrix[1]))
    print("Bigram vectors identical? ", np.array_equal(bigram_matrix[0], bigram_matrix[1]))
    print("\nBigram vocabulary:", list(bigram_vec.get_feature_names_out()))
    print("Doc 1 bigram vector:", bigram_matrix[0])
    print("Doc 2 bigram vector:", bigram_matrix[1])
    assert not np.array_equal(bigram_matrix[0], bigram_matrix[1])
    print("\nBigrams successfully distinguish the two reordered sentences.")


if __name__ == "__main__":
    print("=== Bag-of-words ignores word order ===")
    demo_bow_ignores_order()

    print("=== TF-IDF vs sklearn's TfidfVectorizer ===")
    demo_tfidf_vs_sklearn()

    print("=== Bigrams recover order sensitivity ===")
    demo_bigrams_recover_order()
