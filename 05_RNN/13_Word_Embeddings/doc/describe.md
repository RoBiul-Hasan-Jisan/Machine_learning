# 13. Word Embeddings

## Learning Objectives

- Explain why one-hot encoding is a poor representation for words, and what embeddings fix
- Implement an embedding layer and understand how it's trained jointly with the rest of the network
- Use pretrained embeddings (word2vec/GloVe-style) and understand the analogy properties they exhibit

## The Problem

Lesson 02 mentioned converting text tokens to integer IDs, then treated those IDs as if they were already usable numeric input. They aren't, directly: an integer ID like `"cat" -> 47` carries no information about meaning, and treating IDs as raw numeric input would imply a false ordering (`"cat"=47` is not meaningfully "less than" `"dog"=48`). Word embeddings solve the representation problem this module has been deferring since Lesson 02: how do you turn a discrete token into a numeric vector that a network can meaningfully compute with?

## The Concept

### Why one-hot encoding doesn't work well

The naive fix is **one-hot encoding**: represent word `i` out of a vocabulary of size `V` as a vector of length `V`, all zeros except a 1 at position `i`.

```
Vocabulary: [cat, dog, king, queen, ...]  (size V)

"cat"   -> [1, 0, 0, 0, ...]
"dog"   -> [0, 1, 0, 0, ...]
"king"  -> [0, 0, 1, 0, ...]
```

This avoids the false-ordering problem, but has two serious issues: the vectors are enormous and mostly zeros (for a realistic vocabulary of 50,000+ words, every word is a 50,000-dimensional vector), and — more fundamentally — every pair of distinct words is **equally dissimilar**: the vector for "cat" is exactly as different from "dog" as it is from "king," even though "cat" and "dog" are obviously more related. One-hot encoding has no way to represent that some words are more similar to each other than others.

### Embeddings: dense, learned, meaningful vectors

An **embedding** represents each word as a dense (mostly nonzero), low-dimensional vector (typically 50-300 dimensions, far smaller than the vocabulary size), where the vector's *values* — not just its position — carry meaning, learned from data:

```
"cat"   -> [0.12, -0.43, 0.88, ..., 0.05]     (e.g. 100 real numbers)
"dog"   -> [0.15, -0.39, 0.81, ..., 0.02]     (numerically CLOSE to "cat"'s vector)
"king"  -> [-0.72, 0.34, -0.11, ..., 0.91]    (numerically FAR from "cat"'s vector)
```

An embedding is, mechanically, just a lookup table: a matrix of shape `(vocab_size, embedding_dim)`, where row `i` is word `i`'s vector. Looking up a word's embedding is literally indexing into this matrix — `nn.Embedding` in PyTorch is exactly this lookup table, implemented efficiently.

### Embeddings are learned, not designed

Critically, these vectors aren't hand-specified — they're parameters, initialized randomly (like any other weight matrix) and updated via backpropagation during training, exactly like `W_xh` or an LSTM's gate weights. The embedding table sits as the very first layer of the network (Lesson 03's `x_t` is now, in practice, usually the *output* of an embedding lookup, not a raw one-hot vector), and gradients flow back into it during training just like any other layer.

```python
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=100)

# token_ids: (batch, T) integer tensor of vocabulary indices
embedded = embedding(token_ids)   # (batch, T, 100) -- dense vectors, ready for an RNN
```

As the network trains on its actual task (sentiment classification, translation, whatever the surrounding architecture is built for), the embedding vectors get pushed toward configurations where words that behave similarly in that task end up with similar vectors — nothing about "cat" and "dog" being similar was programmed in; it emerges because they tend to appear in similar contexts relevant to the task.

### Pretrained embeddings

Rather than training embeddings from scratch on a (possibly small) task-specific dataset, it's common to start from embeddings pretrained on a massive text corpus using a dedicated unsupervised objective — **word2vec** (Mikolov et al., 2013) and **GloVe** (Pennington et al., 2014) are the classic examples, trained by predicting a word from its surrounding context (or vice versa) across billions of words of text, entirely without task-specific labels. This is directly analogous to the CNN module's transfer learning lesson: rather than learning representations from scratch on limited task data, start from representations already learned on a much larger, more general dataset, and either freeze them or fine-tune them for the specific task.

```python
# Loading pretrained GloVe vectors into an embedding layer
pretrained_weights = load_glove_vectors(vocab, embedding_dim=100)  # (vocab_size, 100) matrix
embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)  # freeze=True to keep fixed
```

### The famous analogy property

A striking empirical property of word2vec/GloVe-style embeddings: simple vector arithmetic on the learned vectors captures real semantic and syntactic relationships:

```
vector("king") - vector("man") + vector("woman")  ≈  vector("queen")
```

This isn't built into the training objective explicitly — it emerges from the structure the unsupervised training process discovers in how words co-occur across a large corpus. It's a widely cited illustration that these vectors encode genuine relational structure, not just arbitrary numeric labels, and it's a useful sanity check when evaluating a set of trained or pretrained embeddings.

### Embeddings beyond words

The same idea generalizes far beyond words: any discrete, categorical entity (a product ID in a recommendation system, a user ID, an amino acid in a protein sequence, a category label) can be represented with a learned embedding instead of a one-hot vector, for exactly the same reasons — a dense, trainable representation that can capture similarity is almost always preferable to a large sparse one-hot vector once the number of distinct categories grows past a handful.

See `code/embeddings_demo.py` for a from-scratch embedding lookup (implemented as simple matrix indexing, to make the "it's just a lookup table" point concrete), a comparison of one-hot vs embedding representation size, and a demonstration of the analogy property emerging from a small trained embedding.

## Exercises

1. Implement a from-scratch embedding lookup (given a matrix and a list of integer IDs, return the corresponding rows) and verify it matches `nn.Embedding`'s output for the same weight matrix and inputs.
2. Compute the memory footprint (number of floats) of a one-hot representation vs a 100-dimensional embedding for a vocabulary of 50,000 words, and express the difference as a ratio.
3. Train a small embedding layer (as part of a simple classifier, e.g. sentiment classification on short synthetic sentences) and compute cosine similarity between the learned vectors for words that appeared in similar contexts during training vs words that appeared in very different contexts. Confirm the similar-context words end up closer together.
4. Load a small set of pretrained word vectors (or construct a toy embedding table by hand with deliberately chosen values) and check whether `vector("king") - vector("man") + vector("woman")` lands closest to `vector("queen")` among the vocabulary, using cosine similarity or Euclidean distance to the nearest neighbor.

## Key Terms

| Term | What it actually means |
|---|---|
| One-hot encoding | Representing a token as a vector of all zeros except a single 1, with no notion of similarity between different tokens |
| Word embedding | A dense, low-dimensional, learned vector representation of a word, where distance and direction carry meaning |
| Embedding layer | A lookup table (matrix) mapping each vocabulary index to its embedding vector, trained via backpropagation like any other layer |
| Pretrained embeddings | Embedding vectors learned on a large, general text corpus (e.g. via word2vec or GloVe) before being reused or fine-tuned on a specific task |
| word2vec / GloVe | Classic unsupervised methods for learning word embeddings from co-occurrence patterns in large text corpora, without task-specific labels |
