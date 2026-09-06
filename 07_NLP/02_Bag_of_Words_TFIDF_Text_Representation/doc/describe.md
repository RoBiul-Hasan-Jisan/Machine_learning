# 02. Bag of Words, TF-IDF & Text Representation

## Learning Objectives

- Build a bag-of-words representation from scratch and explain what it discards
- Derive and implement TF-IDF, understanding what each of its two factors weights against
- Use n-grams to partially recover the word-order information bag-of-words loses

## The Problem

Lesson 01 turned raw text into tokens. But machine learning models need numeric vectors of a fixed size, not variable-length lists of strings. Turning a document into a fixed-size numeric vector — while preserving as much of the meaningful signal as possible — is the text representation problem, and this lesson covers the classical (pre-embedding) solutions still used throughout information retrieval (Lesson 14) and as strong baselines for classification (Lesson 08).

## The Concept

### Bag of words: count, ignore order

Represent a document as a vector of word counts over a fixed vocabulary, completely discarding word order:

```
Vocabulary: [cat, dog, sat, chased, the]

"the cat sat"        -> [1, 0, 1, 0, 2]
"the dog chased the cat" -> [1, 1, 0, 1, 2]
```

Bag-of-words is simple, fast, and surprisingly effective for tasks like topic classification, where the *presence* of certain words matters more than their order ("basketball", "score", "team" strongly suggest a sports article regardless of sentence structure). It fails badly on tasks where word order changes meaning: "the dog bit the man" and "the man bit the dog" produce the *identical* bag-of-words vector, despite meaning opposite things — the exact limitation that motivated sequential models (the RNN module's Lesson 01 made this same point).

### TF-IDF: weight words by how distinctively informative they are

Raw counts overweight common words ("the", "is", "a") that appear in almost every document and carry little distinguishing information. **Term Frequency-Inverse Document Frequency (TF-IDF)** rescales each count by how *rare* the word is across the whole document collection (corpus), so common words get downweighted and distinctive words get upweighted.

```
TF(word, doc) = count of word in doc / total words in doc
                 (or sometimes just the raw count, depending on convention)

IDF(word, corpus) = log(N / (1 + df(word)))
                     N = total number of documents
                     df(word) = number of documents containing the word

TF-IDF(word, doc) = TF(word, doc) * IDF(word, corpus)
```

Intuition: `IDF` is large for a word that appears in few documents (rare, therefore distinctive when it does appear) and small (approaching 0) for a word that appears in nearly every document (common, therefore uninformative for distinguishing documents from each other). Multiplying by `TF` still rewards a word appearing frequently *within* a specific document, but that reward is scaled down if the word is common *across* the whole corpus.

```
Example: the word "the" appears in every document -> IDF ≈ log(N / N) ≈ 0
         -> TF-IDF("the", any doc) ≈ 0, regardless of how often it appears in that doc

The word "photosynthesis" appears in only 2 of 10,000 documents -> IDF is large
         -> TF-IDF("photosynthesis", a doc containing it) is large,
            correctly flagging it as a distinguishing term for that document
```

### Implementing TF-IDF from scratch

```python
import numpy as np
from collections import Counter

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
```

### N-grams: partially recovering word order

Bag-of-words treats every word independently. **N-grams** extend the vocabulary to include sequences of `n` consecutive words, letting the representation capture some local word-order information without going as far as a fully sequential model:

```
Unigrams (n=1): "the", "dog", "bit", "the", "man"
Bigrams  (n=2): "the dog", "dog bit", "bit the", "the man"
```

A bag-of-*bigrams* representation of "the dog bit the man" and "the man bit the dog" now produces genuinely different vectors — "dog bit" and "bit the man" appear in one but not the other — partially recovering the order sensitivity plain bag-of-words lacks. This comes at a real cost: the vocabulary of possible bigrams is far larger than the vocabulary of unigrams (most word pairs never actually occur, making the representation sparser and higher-dimensional), and this cost grows further with trigrams and beyond. In practice, unigrams + bigrams together are a common, practical middle ground.

### Limitations of count-based representations

Both bag-of-words and TF-IDF share a fundamental limitation regardless of n-gram order: they have no notion that "dog" and "puppy" are related, or that "good" and "great" are near-synonyms — every distinct word gets its own independent dimension, with no encoded similarity between them. This is precisely the gap word embeddings (Lesson 03 onward) are built to close: representing words as dense vectors positioned so that semantically related words end up near each other in the vector space, something no count-based method can provide by construction.

See `code/bow_tfidf_demo.py` for the complete from-scratch TF-IDF implementation verified against `sklearn.feature_extraction.text.TfidfVectorizer`, plus a bigram bag-of-words example showing it distinguishes reordered sentences that plain unigram bag-of-words cannot.

## Exercises

1. Build a bag-of-words matrix for 4 short documents by hand (or with the code) and confirm that two documents using the same words in different orders produce identical vectors.
2. Implement `compute_tfidf` and verify it matches `sklearn`'s `TfidfVectorizer` output (note: sklearn uses a slightly different IDF smoothing formula by default — check its documentation and either match it or explain the difference).
3. Build a bigram bag-of-words representation for "the dog bit the man" and "the man bit the dog" and confirm the two vectors now differ.
4. For a spam classification task and a document-similarity search task, discuss whether TF-IDF or plain bag-of-words counts would likely work better, and why.

## Key Terms

| Term | What it actually means |
|---|---|
| Bag of words | A document representation as a vector of word counts over a fixed vocabulary, discarding word order entirely |
| TF-IDF | A weighting scheme that scales word counts by how rare the word is across the whole document corpus, downweighting common words |
| Term frequency (TF) | How often a word appears within a single document, typically normalized by document length |
| Inverse document frequency (IDF) | A measure of how rare a word is across the corpus, large for rare words and near-zero for words appearing in almost every document |
| N-gram | A sequence of n consecutive tokens, used to partially capture local word order in a count-based representation |
