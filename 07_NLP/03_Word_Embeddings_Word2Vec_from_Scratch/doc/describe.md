# 03. Word Embeddings: Word2Vec from Scratch

## Learning Objectives

- Explain the distributional hypothesis and how it motivates learning embeddings from context
- Implement the Skip-gram architecture with negative sampling from scratch
- Verify that trained embeddings place semantically related words near each other

## The Problem

Lesson 02's bag-of-words and TF-IDF represent every word as an independent dimension — "dog" and "puppy" share no more structure than "dog" and "airplane" do. This is a real, practical problem: a model trained on "the dog is happy" learns nothing that transfers to "the puppy is happy," even though the sentences are nearly synonymous. Word2Vec (Mikolov et al., 2013) fixes this by learning dense vectors where semantically related words end up near each other, purely from patterns in how words are used — no dictionary, no hand-labeling required.

## The Concept

### The distributional hypothesis: "you shall know a word by the company it keeps"

Word2Vec's entire approach rests on one empirical observation: words that appear in similar contexts tend to have similar meanings. "Dog" and "puppy" both frequently appear near words like "bark," "leash," "walk," and "cute" — that shared context pattern is a genuine, learnable signal about their meaning, even though nothing in the raw text explicitly states that dogs and puppies are related. Word2Vec turns this observation directly into a training objective: learn a vector for each word such that it's good at predicting (or being predicted by) its surrounding context words.

### Skip-gram: predict the context from the word

The Skip-gram architecture takes a center word and tries to predict the words that surround it within a fixed window:

```
Sentence: "the quick brown fox jumps over the lazy dog"
Window size 2, center word "fox":

Context words to predict: "quick", "brown", "jumps", "over"

Training pairs generated:  (fox, quick), (fox, brown), (fox, jumps), (fox, over)
```

The network itself is deliberately simple — a single hidden layer with no nonlinearity, more like a factorized lookup table than a deep network:

```
center word (one-hot) -> [Embedding matrix W_in] -> hidden vector (the word's embedding)
                                                            |
                                                            v
                                        [Output matrix W_out] -> scores over the whole vocabulary
                                                            |
                                                            v
                                              softmax -> predicted probability of each context word
```

`W_in` is the matrix whose rows *are* the word embeddings this whole exercise is trying to learn — after training, row `i` of `W_in` is word `i`'s vector. `W_out` is a second set of weights, used only during training to score how well a candidate context word matches, then generally discarded (or occasionally averaged with `W_in`) once training is complete.

### The problem with plain softmax: an enormous output layer

A real vocabulary has tens of thousands to millions of words. Computing a full softmax over the entire vocabulary at every single training step — normalizing a score for every word, most of which are irrelevant to the current example — is prohibitively expensive at that scale, especially since this needs to happen for every one of millions of training pairs across the corpus.

### Negative sampling: turn it into cheap binary classification

Negative sampling reframes the training objective entirely, avoiding the expensive full-vocabulary softmax: instead of predicting "which word out of the entire vocabulary is the correct context word," train a simple binary classifier that answers, separately for each candidate word, "is this word actually a true context word for this center word, or not?"

```
For the pair (fox, brown) [a TRUE context pair from the corpus]:
  Positive example: (fox, brown) -> label 1  (real pair)
  Negative examples: (fox, airplane), (fox, democracy), (fox, teapot) -> label 0
                       (a few random words, sampled as NEGATIVE i.e. NOT true context words)

Loss: binary cross-entropy, pushing the true pair's predicted probability toward 1
      and the random negative pairs' predicted probabilities toward 0
```

Instead of scoring the *entire* vocabulary at every step, negative sampling only scores the one true pair plus a handful (typically 5-20) of randomly sampled negative pairs — turning an expensive `|vocabulary|`-way classification into a cheap handful of binary classifications, while still providing a useful training signal: the embedding gets pushed toward words it truly co-occurs with, and away from words sampled essentially at random.

Negative words are typically sampled with probability proportional to their frequency raised to the 3/4 power (`freq(w)^0.75`), a detail from the original paper that slightly boosts the sampling rate of rare words relative to pure frequency-proportional sampling, giving them more training signal than they'd otherwise get.

### The training objective, concretely

```
sigmoid(v_center . v_context_true)       should be pushed toward 1   (dot product, then squashed to [0,1])
sigmoid(v_center . v_context_negative)   should be pushed toward 0   (for each sampled negative)
```

This dot-product-then-sigmoid structure is why, after training, semantically related words end up with high dot products (and therefore small angles, i.e. pointing in similar directions) in the embedding space: words that truly co-occur are explicitly optimized to have high dot products, and since "dog" and "puppy" co-occur with highly overlapping sets of context words, gradient descent pushes their vectors toward similar directions as a natural consequence of fitting all those overlapping training pairs simultaneously.

### Implementing Skip-gram with negative sampling from scratch

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_step(center_idx, context_idx, negative_indices, W_in, W_out, lr):
    v_center = W_in[center_idx]                    # (embedding_dim,)

    # Positive pair
    v_context = W_out[context_idx]
    score = sigmoid(v_center @ v_context)
    grad = (score - 1)                              # gradient of the loss w.r.t. the dot product
    W_out[context_idx] -= lr * grad * v_center
    grad_center_total = grad * v_context

    # Negative pairs
    for neg_idx in negative_indices:
        v_neg = W_out[neg_idx]
        score_neg = sigmoid(v_center @ v_neg)
        grad_neg = score_neg                         # target is 0, so gradient is just the score
        W_out[neg_idx] -= lr * grad_neg * v_center
        grad_center_total += grad_neg * v_neg

    W_in[center_idx] -= lr * grad_center_total
```

See `code/word2vec_demo.py` for the complete Skip-gram implementation trained on a small synthetic corpus with clear semantic clusters, showing (via cosine similarity) that the trained embeddings correctly place related words near each other and unrelated words far apart — the direct, measurable payoff of the distributional hypothesis.

## Exercises

1. Generate training pairs (center word, context word) from a sentence using a window size of 2, then repeat with window size 4. Explain how a larger window changes what kind of relationships the embeddings tend to capture.
2. Implement `train_step` and train it on a small synthetic corpus for several epochs. Confirm that words appearing in similar contexts end up with high cosine similarity.
3. Implement frequency^0.75-based negative sampling and compare the resulting negative sample distribution to pure frequency-proportional sampling on a corpus with a few very common words and many rare ones.
4. Using your trained embeddings, perform the classic "king - man + woman ≈ queen" style vector arithmetic on a synthetic corpus designed to contain an analogous relationship, and check whether the nearest neighbor of the resulting vector is the expected word.

## Key Terms

| Term | What it actually means |
|---|---|
| Distributional hypothesis | The idea that words appearing in similar contexts tend to have similar meanings, the foundation of word embedding methods |
| Skip-gram | A Word2Vec architecture that trains embeddings by predicting a word's surrounding context words |
| Negative sampling | A training technique that replaces an expensive full-vocabulary softmax with a cheap binary classification between true and randomly sampled false context pairs |
| Word embedding | A dense, learned vector representation of a word, positioned so that semantically related words are near each other in the vector space |
