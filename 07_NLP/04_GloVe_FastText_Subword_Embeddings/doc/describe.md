# 04. GloVe, FastText & Subword Embeddings

## Learning Objectives

- Explain how GloVe's co-occurrence-matrix approach differs from Word2Vec's prediction-based approach
- Explain how FastText represents words as bags of character n-grams and why this handles rare and unseen words
- Implement a simplified GloVe training objective and a subword embedding lookup from scratch

## The Problem

Word2Vec (Lesson 03) learns embeddings by predicting context words from a stream of individual training pairs, one pair at a time — it never directly looks at global corpus statistics, only local windows. It also assigns exactly one vector per whole word, which fails for any word not seen during training (out-of-vocabulary, or OOV) and can't exploit the fact that "run," "runs," "running," and "runner" obviously share meaning through their shared root, since Word2Vec treats each as a completely independent vocabulary entry. GloVe and FastText each address one of these two limitations.

## The Concept

### GloVe: learn from global co-occurrence statistics directly

GloVe ("Global Vectors," Pennington et al., 2014) starts by building a full word-word co-occurrence matrix across the entire corpus — a count, for every pair of words, of how often they appear near each other — and then learns embeddings whose dot products directly approximate the *log* of those co-occurrence counts.

```
Co-occurrence matrix X: X[i][j] = number of times word j appears in word i's context window,
                                    counted across the WHOLE corpus (not one training pair at a time)

Training objective (simplified):

minimize  sum over all word pairs (i, j) with X[i][j] > 0 of:
    f(X[i][j]) * (v_i . v_j + b_i + b_j - log(X[i][j]))^2

f(X[i][j]) is a weighting function that down-weights very frequent pairs
(like "the", "of") so they don't dominate the loss, similar in spirit to
TF-IDF's downweighting of common words (Lesson 02)
```

Where Word2Vec sees the corpus as a stream of local prediction problems, GloVe sees it as one global matrix-factorization problem: find vectors `v_i` such that their dot products reconstruct the (log) co-occurrence counts as accurately as possible. In practice, GloVe and Word2Vec tend to produce embeddings of broadly similar quality on similar tasks — the choice between them is less about accuracy and more about training characteristics (GloVe's global matrix approach can be more efficient to train on very large corpora since co-occurrence statistics are computed once, rather than iterating over every individual context window repeatedly) and availability of high-quality pretrained vectors, which GloVe made freely available early and widely.

### The shared limitation: whole-word vectors

Both Word2Vec and GloVe assign one fixed vector per whole word in the training vocabulary. Two direct consequences:

- **Out-of-vocabulary (OOV) words get no vector at all.** A word never seen during training — a typo, a brand-new term, a rare proper noun — has no representation, forcing some fallback (a generic "unknown" vector, or dropping the word entirely).
- **No sharing between morphologically related words.** "Run," "runs," "running," and "runner" are trained as four entirely independent vectors, with nothing in the model architecture encouraging them to end up related, even though their shared root is an obvious, exploitable signal a human reader uses instantly.

### FastText: represent a word as a bag of its character n-grams

FastText (Bojanowski et al., 2016, from the same research group as Word2Vec) fixes both problems with one idea: represent each word not as a single atomic vector, but as a **pooled combination of the vectors of its character n-grams**.

```
Word "running", with character n-grams of length 3-6, marked with boundary symbols:

<ru, run, unn, nni, nin, ing, ng>, <run, runn, unni, nnin, ning, ing>, ...
(plus the special whole-word token <running>)

Word vector = combination (sum or mean) of the vectors of ALL these n-gram tokens
```

Each character n-gram gets its own trainable vector (following the same Skip-gram-with-negative-sampling training procedure as Word2Vec, Lesson 03, just applied to n-gram tokens rather than whole-word tokens), and a word's final vector pools all its constituent n-grams' vectors together. This directly solves both problems:

- **Morphologically related words share n-grams**, and therefore share components of their vectors automatically — "running" and "runner" both contain the n-gram "run" and others, so their vectors are related by construction, not just by learned co-occurrence patterns.
- **OOV words still get a vector.** Even a word never seen during training can be broken into character n-grams, most of which likely *were* seen (as substrings of other training words), letting FastText compute a reasonable vector for it on the fly.

Note: this is the standard textbook formula (following the original paper), with input and hidden weights combined into one matrix per gate for readability. Production implementations differ in two small but real ways worth knowing about if you ever compare against them directly: real FastText training uses hashing to bound the number of distinct n-gram vectors in memory (rather than one vector per unique n-gram string), and it typically pools n-grams by mean rather than raw sum so a word's vector magnitude doesn't grow with its length. The code demo below implements mean pooling for exactly this reason.

```python
def get_fasttext_vector(word, ngram_vectors, min_n=3, max_n=6):
    word_bounded = f"<{word}>"
    ngrams = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(word_bounded) - n + 1):
            ngrams.add(word_bounded[i:i + n])
    ngrams.add(word_bounded)  # the whole word itself, as a special token

    vectors = [ngram_vectors[g] for g in ngrams if g in ngram_vectors]
    if not vectors:
        return None  # every single n-gram is also unseen -- genuinely no signal available
    return np.mean(vectors, axis=0)
```

### Subword embeddings beyond FastText

FastText's character-n-gram idea is a specific instance of a broader pattern: **subword tokenization**, which breaks words into smaller learned units before any embedding is computed at all, rather than embedding whole words and only falling back to subwords for OOV cases. This is the approach essentially every modern deep learning NLP model uses — Lesson 19 covers the specific algorithms (BPE, WordPiece, Unigram, SentencePiece) that learn which subword units to use directly from a training corpus, and those learned subword embeddings are what feeds into the Transformer-based architectures used throughout the rest of this module.

### Comparing the three approaches

| | Word2Vec | GloVe | FastText |
|---|---|---|---|
| Training signal | Local context windows, one pair at a time | Global co-occurrence matrix | Local context windows, on character n-grams |
| Handles OOV words | No | No | Yes, via shared n-grams |
| Captures morphology | No | No | Yes, by construction |
| Vector granularity | Whole word | Whole word | Character n-gram, pooled per word |

See `code/glove_fasttext_demo.py` for a simplified from-scratch GloVe training loop on a small co-occurrence matrix, and a real (small-scale) FastText training loop demonstrating both morphological similarity and OOV word handling.

## Exercises

1. Build a co-occurrence matrix by hand for a 4-sentence toy corpus with a window size of 2, and identify which word pairs have the highest co-occurrence counts.
2. Implement the simplified GloVe training loop and confirm that frequently co-occurring word pairs end up with higher dot products than rarely co-occurring pairs.
3. Implement `get_fasttext_vector` and compute the cosine similarity between "running" and "runner" using shared n-grams, versus "running" and a word sharing a different root.
4. Compute a FastText-style vector for a word that was never seen during training (but shares n-grams with trained words) and confirm it produces a non-trivial, non-random vector rather than failing outright.

## Key Terms

| Term | What it actually means |
|---|---|
| GloVe | A word embedding method that learns vectors by factorizing a global word-word co-occurrence matrix, rather than predicting local context one window at a time |
| Co-occurrence matrix | A matrix recording how often each pair of words appears near each other across an entire corpus |
| FastText | A word embedding method representing each word as a pooled combination of vectors for its character n-grams, enabling morphological sharing and OOV handling |
| Out-of-vocabulary (OOV) | A word not present in a model's training vocabulary, which whole-word embedding methods like Word2Vec and GloVe cannot represent at all |
| Character n-gram | A contiguous sequence of n characters within a word, the basic unit FastText embeds and pools to form word vectors |
