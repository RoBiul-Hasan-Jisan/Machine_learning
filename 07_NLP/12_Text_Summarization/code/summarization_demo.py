"""
A from-scratch TextRank extractive summarizer using TF-IDF sentence
similarity, and a from-scratch ROUGE-N / ROUGE-L implementation for
evaluation.
"""

import re
from collections import Counter

import numpy as np


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def compute_tfidf_vectors(sentences):
    tokenized = [tokenize(s) for s in sentences]
    vocab = sorted(set(w for toks in tokenized for w in toks))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    N = len(sentences)
    df = Counter()
    for toks in tokenized:
        for w in set(toks):
            df[w] += 1

    idf = {w: np.log(N / (1 + df[w])) for w in vocab}

    vectors = np.zeros((N, len(vocab)))
    for i, toks in enumerate(tokenized):
        counts = Counter(toks)
        total = len(toks) if toks else 1
        for w, c in counts.items():
            vectors[i, word_to_idx[w]] = (c / total) * idf[w]
    return vectors


def cosine_similarity_matrix(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = vectors / norms
    return normalized @ normalized.T


def textrank_scores(similarity_matrix, damping=0.85, n_iter=50):
    n = similarity_matrix.shape[0]
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = similarity_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(n_iter):
        scores = (1 - damping) / n + damping * (transition.T @ scores)
    return scores


def textrank_summarize(text, n_sentences=2):
    sentences = split_sentences(text)
    if len(sentences) <= n_sentences:
        return sentences, np.ones(len(sentences))

    vectors = compute_tfidf_vectors(sentences)
    sim_matrix = cosine_similarity_matrix(vectors)
    np.fill_diagonal(sim_matrix, 0)

    scores = textrank_scores(sim_matrix)
    ranked_idx = np.argsort(-scores)[:n_sentences]
    ranked_idx = sorted(ranked_idx)

    return [sentences[i] for i in ranked_idx], scores


def demo_textrank():
    document = (
        "The company reported strong Q3 earnings, beating analyst expectations by 15 percent. "
        "Revenue grew 20 percent year over year, driven primarily by cloud services. "
        "The CEO cited strong enterprise demand as the key driver of growth. "
        "The company also announced a new office opening in Austin next year. "
        "Employees will have the option to relocate or work remotely. "
        "Analysts noted that cloud revenue has been the company's fastest-growing segment for six straight quarters. "
        "The stock price rose 8 percent following the earnings announcement."
    )

    sentences = split_sentences(document)
    print(f"Document has {len(sentences)} sentences.\n")

    summary, scores = textrank_summarize(document, n_sentences=3)

    print("=== Sentence importance scores ===")
    for s, score in zip(sentences, scores):
        print(f"  [{score:.4f}] {s}")

    print("\n=== TextRank extractive summary (top 3 sentences, original order) ===")
    for s in summary:
        print(f"  - {s}")


def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_recall(candidate, reference, n=1):
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    if not ref_ngrams:
        return 0.0
    overlap = sum(min(count, cand_ngrams[gram]) for gram, count in ref_ngrams.items())
    total_ref = sum(ref_ngrams.values())
    return overlap / total_ref


def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_recall(candidate, reference):
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return 0.0
    lcs = lcs_length(cand_tokens, ref_tokens)
    return lcs / len(ref_tokens)


def demo_rouge():
    reference = "the company reported strong earnings driven by cloud revenue growth"
    candidates = [
        "the company reported strong earnings driven by cloud revenue growth",
        "strong earnings were reported by the company thanks to cloud revenue growth",
        "the stock price rose after the announcement",
    ]

    print("Reference:", reference)
    for c in candidates:
        r1 = rouge_n_recall(c, reference, n=1)
        rl = rouge_l_recall(c, reference)
        print(f"\n  candidate: '{c}'")
        print(f"    ROUGE-1 recall: {r1:.4f}")
        print(f"    ROUGE-L recall: {rl:.4f}")

    print("\nNote how the reordered/paraphrased candidate keeps a high ROUGE-1 recall")
    print("(same words present) but a lower ROUGE-L recall (the word ORDER/sequence")
    print("structure matches less closely) -- the two metrics capture different")
    print("aspects of overlap with the reference.")


if __name__ == "__main__":
    print("=== TextRank extractive summarization ===")
    demo_textrank()

    print("\n=== ROUGE evaluation ===")
    demo_rouge()
