# 02. Sequential Data and Memory

## Learning Objectives

- Represent text, time series, and categorical sequences as numeric tensors an RNN can consume
- Explain padding, truncation, and masking for handling variable-length sequences in batches
- Distinguish what "memory" means in an RNN from a literal buffer of past inputs

## The Problem

Lesson 01 said an RNN processes "a sequence" — but a sequence of what, represented how? Raw text is a string of characters. A time series is a list of numbers with timestamps. Neither is directly usable by a network, which needs numeric tensors of consistent shape. This lesson covers turning real sequential data into that form, and what "the network remembers" concretely refers to once it's in that form.

## The Concept

### Representing a sequence as a tensor

A sequence of length `T`, where each element is a vector of size `D` (a word's embedding, a time series' feature vector, etc.), is represented as a tensor of shape `(T, D)` for one example, or `(batch, T, D)` for a batch of examples processed together (matching the batch convention introduced for CNNs). Concretely:

```
Text:  "the cat sat"  ->  tokenize  ->  [id_the, id_cat, id_sat]  ->  embed each id (Lesson 13)
                                          -> tensor of shape (3, embedding_dim)

Time series:  [23.1, 24.0, 22.8, 25.3]  ->  reshape each scalar into a 1-dim feature
                                          -> tensor of shape (4, 1)

Multivariate time series (e.g. daily temperature + humidity):
    [[23.1, 60], [24.0, 58], [22.8, 65]]  ->  tensor of shape (3, 2)
```

Text requires an extra step — **tokenization** (splitting text into discrete units: words, subwords, or characters) and **numericalization** (mapping each token to an integer ID via a fixed vocabulary) — before the sequence even has numeric values to arrange into a tensor. Lesson 13 covers turning those integer IDs into dense, meaningful vectors (embeddings); for now, treat each token as just an ID.

### Batching sequences of different lengths

Real datasets rarely have every sequence be exactly the same length — sentences vary, time series windows vary. Since a batch tensor needs a single, consistent shape, shorter sequences are extended to match the longest sequence in the batch (or a fixed maximum length) using **padding**, typically a reserved "pad" token (often ID 0) or the value 0 for numeric data.

```
Batch of 3 sentences, as token IDs, padded to the longest (length 5):

[ 4, 17, 9, 0, 0]     <- length 3, padded with 2 zeros
[ 8,  2, 5, 11, 3]    <- length 5, no padding needed
[19, 6, 0, 0, 0]      <- length 2, padded with 3 zeros

Shape: (batch=3, T=5)
```

Padding solves the shape problem but introduces a new one: the network shouldn't treat pad tokens as real data. A **mask** — a same-shaped tensor of 1s (real data) and 0s (padding) — tells downstream computations (loss functions, attention in Lesson 12) to ignore the padded positions.

```
Mask for the batch above:

[1, 1, 1, 0, 0]
[1, 1, 1, 1, 1]
[1, 1, 0, 0, 0]
```

Alternatively (and often more efficiently), sequences within a batch can be sorted by length and processed with `pack_padded_sequence` (PyTorch's mechanism for skipping computation on padded positions entirely, rather than computing on them and then masking the result) — the from-scratch demo below covers the manual masking approach since it makes the mechanics explicit; the "Use It" section shows the more efficient built-in tool.

### Truncation

Very long sequences (a full document, a long audio clip) are often truncated to a maximum length for practical reasons: computational cost per sequence scales with length (relevant again in Lesson 05's backpropagation-through-time cost), and extremely long sequences make certain gradient problems (Lesson 06) worse. Truncation is a real, sometimes costly tradeoff — truncating a document classification task to its first 200 words discards potentially relevant information later in the document — and the right maximum length is a genuine design decision, not just an implementation detail to set once and ignore.

### What "memory" actually means here

It's tempting to picture an RNN's hidden state as literally storing past inputs, like a buffer or a list. It doesn't. The hidden state is a fixed-size vector (say, 128 numbers) that gets *overwritten* at every step by a function of the current input and the previous hidden state (Lesson 03 defines this function exactly). Nothing about the mechanism guarantees any particular piece of information from step 3 will still be recoverable from the hidden state at step 50 — that's an empirical property of what the network learns to preserve during training, not something built into the architecture by construction. This distinction matters: it's exactly the reason plain RNNs struggle with long-range dependencies (Lesson 06), and exactly the problem LSTM and GRU (Lessons 07-08) are specifically designed to address.

See `code/sequence_data_demo.py` for tokenizing and numericalizing text, padding a batch of variable-length sequences, constructing a mask, and a comparison of manual masking vs `pack_padded_sequence`.

## Exercises

1. Tokenize three sentences of different lengths (splitting on whitespace is enough), build a small vocabulary, and convert each sentence to a list of integer IDs.
2. Pad the three tokenized sentences from exercise 1 to a common length and construct the corresponding mask tensor.
3. Given a batch of padded sequences and their mask, compute the average sequence length in the batch using only the mask (not the original lengths), to confirm you understand what information the mask actually encodes.
4. Using PyTorch's `nn.utils.rnn.pack_padded_sequence`, process a padded batch through an `nn.RNN` layer and confirm the packed version's output differs from naively running the RNN on the padded tensor without masking.

## Key Terms

| Term | What it actually means |
|---|---|
| Tokenization | Splitting raw text into discrete units (words, subwords, or characters) before converting them to numeric IDs |
| Padding | Extending shorter sequences in a batch with a filler value (often 0) so every sequence in the batch has the same length |
| Mask | A same-shaped tensor of 1s and 0s marking which positions are real data versus padding, used to make downstream computations ignore padded positions |
| Truncation | Cutting a sequence down to a maximum length, discarding the excess |
| pack_padded_sequence | A PyTorch utility that lets an RNN skip computation on padded positions directly, rather than computing on them and masking afterward |
