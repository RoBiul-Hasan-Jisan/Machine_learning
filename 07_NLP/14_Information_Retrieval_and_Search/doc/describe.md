# 14. Information Retrieval & Search

## Learning Objectives

- Implement an inverted index and explain why it makes large-scale search tractable
- Implement BM25 ranking and explain how it improves on plain TF-IDF for retrieval
- Distinguish sparse (lexical) retrieval from dense (embedding-based) retrieval

## The Problem

Given a query and a large collection of documents, find the documents most relevant to that query, ranked by relevance — the core problem behind every search engine, and, as Lesson 13 established, a required component of any open-domain QA or retrieval-augmented system. Doing this efficiently at scale (millions or billions of documents) and effectively (actually surfacing the right documents) requires specific data structures and ranking functions beyond just "compute TF-IDF similarity to everything," which becomes far too slow once the collection is large.

## The Concept

### The inverted index: making search tractable at scale

Computing similarity between a query and *every* document in a large collection (a "forward" scan) doesn't scale — checking millions of documents for every single query is far too slow for real-time search. An **inverted index** flips the data structure: instead of documents mapping to their words, it stores, for every word, the list of documents containing it.

```
Forward index (document -> words):              Inverted index (word -> documents):
  doc1: ["cat", "sat", "mat"]                      "cat":  [doc1, doc3]
  doc2: ["dog", "ran", "park"]                      "sat":  [doc1]
  doc3: ["cat", "ran", "fast"]                       "mat":  [doc1]
                                                       "dog":  [doc2]
                                                       "ran":  [doc2, doc3]
                                                       "park": [doc2]
                                                       "fast": [doc3]
```

To answer a query, look up each query term's list of containing documents directly (a fast hash-table lookup) instead of scanning every document — for a query like "cat ran," only `doc1, doc3` (containing "cat") and `doc2, doc3` (containing "ran") ever need to be considered at all, letting a search engine skip the overwhelming majority of a large collection entirely rather than scoring every single document against the query.

```python
def build_inverted_index(documents):
    index = {}
    for doc_id, tokens in enumerate(documents):
        for token in set(tokens):
            index.setdefault(token, []).append(doc_id)
    return index

def candidate_documents(query_tokens, index):
    candidates = set()
    for token in query_tokens:
        candidates.update(index.get(token, []))
    return candidates
```

### BM25: a stronger ranking function than plain TF-IDF

TF-IDF (Lesson 02) is a reasonable relevance signal, but BM25 (Best Matching 25, from the 1990s Okapi retrieval system) is the long-standing, more effective standard for ranking retrieved documents, refining TF-IDF's core idea with two specific improvements:

```
BM25(query, doc) = sum over query terms t of:
    IDF(t) * ( TF(t, doc) * (k1 + 1) ) / ( TF(t, doc) + k1 * (1 - b + b * |doc| / avgdl) )

k1: controls how quickly term frequency saturates (typically ~1.2-2.0)
b:  controls how much document length is penalized (typically ~0.75)
avgdl: the average document length across the whole collection
```

Two concrete improvements over plain TF-IDF:

- **Term frequency saturation.** Plain TF scales linearly — a word appearing 20 times counts as "twice as relevant" as appearing 10 times, forever. BM25's `TF * (k1+1) / (TF + k1)` term instead saturates: after a term has appeared several times, additional repetitions add diminishing marginal relevance, matching the practical intuition that a document mentioning "python" 20 times isn't meaningfully "more about python" than one mentioning it 10 times — the term has already made its relevance clear well before then.
- **Document length normalization.** A longer document tends to contain more instances of any given term simply by virtue of having more words overall, not because it's more relevant. The `|doc| / avgdl` term explicitly corrects for this, penalizing raw term counts in unusually long documents so length alone doesn't inflate a document's score.

```python
import numpy as np

def bm25_score(query_tokens, doc_tokens, all_docs_tokens, k1=1.5, b=0.75):
    avgdl = np.mean([len(d) for d in all_docs_tokens])
    doc_len = len(doc_tokens)
    N = len(all_docs_tokens)

    score = 0.0
    for term in query_tokens:
        tf = doc_tokens.count(term)
        df = sum(1 for d in all_docs_tokens if term in d)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
        score += idf * numerator / denominator
    return score
```

BM25 remains a strong, fast, and widely-used baseline — many production search systems still use it directly, or use it as one signal combined with others (including the dense retrieval methods below).

### Sparse vs dense retrieval

Everything so far in this lesson — the inverted index, TF-IDF, BM25 — is **sparse (lexical) retrieval**: matching is based on exact word overlap between query and document, with each word an independent dimension (Lesson 02's bag-of-words limitation, inherited directly). This means a query for "automobile" won't match a document that only says "car," even though they're synonyms, unless the exact word appears.

**Dense retrieval** instead represents both queries and documents as dense embedding vectors (Lesson 22 covers modern embedding models used for exactly this purpose) and ranks documents by embedding similarity (typically cosine similarity or dot product) rather than exact word overlap — this captures semantic relevance ("automobile" and "car" end up with similar embeddings, per the distributional hypothesis from Lesson 03) that sparse methods structurally cannot.

| | Sparse (BM25/TF-IDF) | Dense (embedding-based) |
|---|---|---|
| Matching basis | Exact word overlap | Semantic/embedding similarity |
| Handles synonyms | No | Yes |
| Handles exact phrase/keyword matches | Very well (a real strength) | Can miss exact rare terms embeddings weren't trained to distinguish sharply |
| Compute cost per query | Low (inverted index lookup) | Higher (embedding computation + nearest-neighbor search, though approximate nearest-neighbor structures make this practical at scale) |
| Common practice | Still a strong standalone baseline | Often combined with sparse retrieval (hybrid search) rather than used alone |

In practice, **hybrid search** — combining sparse and dense retrieval scores, since each catches cases the other misses — is increasingly the standard in production retrieval-augmented systems, precisely because sparse retrieval's exact-match strength and dense retrieval's semantic-match strength are complementary rather than redundant.

See `code/ir_demo.py` for a from-scratch inverted index and BM25 implementation, plus a side-by-side comparison against TF-IDF cosine similarity ranking and a simple dense-embedding-based retrieval comparison illustrating the synonym-matching gap.

## Exercises

1. Build an inverted index for a small document collection and use it to quickly find candidate documents for a multi-word query, confirming it correctly skips documents containing none of the query terms.
2. Implement `bm25_score` and rank a small document collection for a test query, comparing the ranking order against plain TF-IDF cosine similarity ranking (Lesson 02) on the same collection.
3. Construct a query containing a synonym of a word that appears in a relevant document but not the exact query word (e.g. query "automobile," document contains "car"). Confirm BM25/TF-IDF retrieval misses it, then confirm a word-embedding-based similarity (Lesson 03/04) can retrieve it.
4. Experiment with BM25's `k1` and `b` parameters on a small collection with documents of varying length, and describe how changing `b` from 0 to 1 affects the ranking of a long document that repeats the query term many times.

## Key Terms

| Term | What it actually means |
|---|---|
| Inverted index | A data structure mapping each term to the list of documents containing it, enabling fast candidate lookup instead of scanning the full collection |
| BM25 | A ranking function refining TF-IDF with term frequency saturation and document length normalization, a long-standing strong baseline for search |
| Term frequency saturation | The property that additional occurrences of a term contribute diminishing marginal relevance score, rather than scaling linearly forever |
| Sparse (lexical) retrieval | Retrieval based on exact word overlap between query and document |
| Dense retrieval | Retrieval based on embedding similarity, capturing semantic relevance beyond exact word matches |
| Hybrid search | Combining sparse and dense retrieval signals, since each catches relevant cases the other misses |
