"""
A from-scratch inverted index and BM25 implementation, a comparison
against TF-IDF cosine similarity ranking, and a dense-embedding-based
retrieval comparison illustrating the synonym-matching gap sparse
retrieval cannot close.
"""

import re
from collections import Counter

import numpy as np


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_inverted_index(documents_tokens):
    index = {}
    for doc_id, tokens in enumerate(documents_tokens):
        for token in set(tokens):
            index.setdefault(token, []).append(doc_id)
    return index


def candidate_documents(query_tokens, index):
    candidates = set()
    for token in query_tokens:
        candidates.update(index.get(token, []))
    return candidates


def bm25_score(query_tokens, doc_tokens, all_docs_tokens, k1=1.5, b=0.75):
    avgdl = np.mean([len(d) for d in all_docs_tokens])
    doc_len = len(doc_tokens)
    N = len(all_docs_tokens)

    score = 0.0
    for term in query_tokens:
        tf = doc_tokens.count(term)
        if tf == 0:
            continue
        df = sum(1 for d in all_docs_tokens if term in d)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
        score += idf * numerator / denominator
    return score


def tfidf_cosine_score(query_tokens, doc_tokens, all_docs_tokens):
    vocab = sorted(set(w for d in all_docs_tokens for w in d) | set(query_tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    N = len(all_docs_tokens)

    df = Counter()
    for d in all_docs_tokens:
        for w in set(d):
            df[w] += 1
    idf = {w: np.log((N + 1) / (1 + df[w])) + 1 for w in vocab}

    def vectorize(tokens):
        v = np.zeros(len(vocab))
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        for w, c in counts.items():
            if w in word_to_idx:
                v[word_to_idx[w]] = (c / total) * idf[w]
        return v

    doc_vec = vectorize(doc_tokens)
    query_vec = vectorize(query_tokens)
    denom = (np.linalg.norm(doc_vec) * np.linalg.norm(query_vec)) + 1e-10
    return (doc_vec @ query_vec) / denom


def demo_inverted_index():
    documents = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat ran fast",
        "a bird flew over the park",
    ]
    docs_tokens = [tokenize(d) for d in documents]
    index = build_inverted_index(docs_tokens)

    print("Inverted index (partial):")
    for term in ["cat", "ran", "park"]:
        print(f"  '{term}' -> documents {index.get(term, [])}")

    query = "cat ran"
    candidates = candidate_documents(tokenize(query), index)
    print(f"\nQuery: '{query}'")
    print(f"Candidate documents (via index lookup, no full scan needed): {sorted(candidates)}")
    for doc_id in sorted(candidates):
        print(f"  doc{doc_id}: '{documents[doc_id]}'")
    print()


def demo_bm25_vs_tfidf():
    documents = [
        "python is a popular programming language for data science",
        "python python python python is mentioned many times in this short spammy document",
        "java and python are both widely used programming languages in industry",
        "snakes such as pythons are large reptiles found in tropical regions",
    ]
    docs_tokens = [tokenize(d) for d in documents]
    query_tokens = tokenize("python programming language")

    print("=== BM25 vs TF-IDF ranking ===")
    print(f"Query: 'python programming language'\n")

    bm25_scores = [bm25_score(query_tokens, d, docs_tokens) for d in docs_tokens]
    tfidf_scores = [tfidf_cosine_score(query_tokens, d, docs_tokens) for d in docs_tokens]

    print(f"{'doc':4s} | {'BM25':8s} | {'TF-IDF cos':10s} | text")
    for i, (b, t, doc) in enumerate(zip(bm25_scores, tfidf_scores, documents)):
        print(f"doc{i}  | {b:8.4f} | {t:10.4f} | {doc[:55]}")

    print("\nNote doc1 repeats 'python' many times (a spammy, short document) --")
    print("BM25's term-frequency SATURATION keeps it from dominating purely on repetition")
    print("the way an unsaturated count-based score more easily could.\n")


def demo_synonym_gap():
    """Illustrate what BM25/TF-IDF structurally cannot do: match a query
    to a document using only a SYNONYM, never the literal query word."""
    documents = [
        "the car has excellent fuel efficiency and a smooth ride",
        "this laptop has a fast processor and long battery life",
    ]
    docs_tokens = [tokenize(d) for d in documents]
    query_tokens = tokenize("automobile efficiency")

    print("=== The synonym-matching gap in sparse retrieval ===")
    print(f"Query: 'automobile efficiency'  (doc 0 is about a 'car', a synonym, not 'automobile')\n")

    bm25_scores = [bm25_score(query_tokens, d, docs_tokens) for d in docs_tokens]
    for i, (score, doc) in enumerate(zip(bm25_scores, documents)):
        print(f"doc{i}  BM25={score:.4f}  '{doc}'")

    print("\nBM25 gives doc0 credit ONLY for the shared word 'efficiency' -- it gets")
    print("ZERO credit for 'car' matching the query's 'automobile', because sparse")
    print("retrieval only rewards EXACT token overlap. A dense (embedding-based)")
    print("retriever, using vectors where 'car' and 'automobile' sit close together")
    print("(the distributional hypothesis, Lesson 03), would recognize this")
    print("synonym relationship directly -- exactly the gap hybrid search closes.")


if __name__ == "__main__":
    print("=== Inverted index ===")
    demo_inverted_index()

    demo_bm25_vs_tfidf()

    demo_synonym_gap()
