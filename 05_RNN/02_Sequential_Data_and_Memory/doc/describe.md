# Sequential Data and Memory



##  The Problem: Raw Sequences Aren't Numbers Yet

An RNN processes "a sequence" — but a sequence of *what*, represented *how*? This turns out to matter a lot in practice:

- Raw text is a string of characters: `"the cat sat"`.
- A time series is a list of numbers attached to timestamps: `[23.1, 24.0, 22.8, 25.3]`.
- A categorical sequence might be a list of labels: `["Monday", "Tuesday", "Wednesday"]`.

None of these is directly usable by a network. A network needs **numeric tensors of consistent shape** — the same requirement a CNN has for images, just applied along a time axis instead of a spatial grid. This lesson is about closing that gap: turning real, messy sequential data into tensors, and then being precise about what the resulting "memory" actually is.

---

##  Representing a Sequence as a Tensor

###  The target shape

A sequence of length $T$, where each element is a vector of size $D$, is represented as a tensor of shape:

$$(T, D) \quad \text{for one example, or} \quad (\text{batch}, T, D) \quad \text{for a batch}$$

This mirrors the batch convention from the CNN module, just with a time axis $T$ in place of spatial height/width axes. Two concrete cases:

```
Text:  "the cat sat"  ->  tokenize  ->  [id_the, id_cat, id_sat]  ->  embed each id (Lesson 13)
                                          -> tensor of shape (3, embedding_dim)

Time series:  [23.1, 24.0, 22.8, 25.3]  ->  reshape each scalar into a 1-dim feature
                                          -> tensor of shape (4, 1)

Multivariate time series (e.g. daily temperature + humidity):
    [[23.1, 60], [24.0, 58], [22.8, 65]]  ->  tensor of shape (3, 2)
```

Here, $T$ is the sequence length (3 words, 4 days) and $D$ is the feature dimension per step (an embedding size, or 1 number, or 2 numbers). Getting comfortable reading "$(T, D)$" and immediately knowing which axis is which will save you real debugging time later — a shape mismatch between $T$ and $D$ is one of the most common bugs when building RNNs.

### Turning text into numbers: tokenization and numericalization

Text needs two extra steps before it has any numbers to arrange into a tensor at all:

**Step 1 — Tokenization**: splitting text into discrete units. The simplest choice is splitting on whitespace (word-level tokens); more sophisticated schemes split into subwords or characters (used by most modern language models, but out of scope here).

**Step 2 — Numericalization**: mapping each token to an integer ID using a fixed vocabulary (a lookup table built once from your training data).

Worked example:

```
Vocabulary (built from a training set):
  {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "dog": 4, "ran": 5}

Sentence: "the cat sat"
Step 1 (tokenize):     ["the", "cat", "sat"]
Step 2 (numericalize): [1, 2, 3]
```

Each integer ID is just an arbitrary label at this point — ID `2` isn't "closer" to ID `3` in any meaningful sense yet. Lesson 13 covers **embeddings**: a learned mapping from each integer ID to a dense vector that *does* carry meaning (so that, e.g., "cat" and "dog" end up with similar vectors). For this lesson, treat each token as just an ID — the tensor shape $(T,)$ of integer IDs, before embedding, or $(T, D)$ after embedding into $D$-dimensional vectors.

###  Batching sequences of different lengths

Real datasets rarely have every sequence be exactly the same length. But a batch tensor needs one consistent shape across all examples in the batch — you can't stack a length-3 sequence and a length-5 sequence into a single rectangular tensor without doing *something* about the mismatch. That something is **padding**: extending shorter sequences with a filler value (typically a reserved pad token, ID `0` above) until every sequence in the batch matches the length of the longest one.

```
Batch of 3 sentences, as token IDs, padded to the longest (length 5):

[ 4, 17, 9, 0, 0]     <- length 3, padded with 2 zeros
[ 8,  2, 5, 11, 3]    <- length 5, no padding needed
[19, 6, 0, 0, 0]      <- length 2, padded with 3 zeros

Shape: (batch=3, T=5)
```

###  Masks: telling the network what's real

Padding solves the *shape* problem, but creates a new one: nothing so far stops the network from treating those padding zeros as if they were real data — running them through the RNN update equation from Lesson 01 exactly like any other input. A **mask** fixes this: a same-shaped tensor of 1s (real data) and 0s (padding), used by downstream computations (loss functions, attention in Lesson 12) to *ignore* the padded positions.

```
Mask for the batch above:

[1, 1, 1, 0, 0]
[1, 1, 1, 1, 1]
[1, 1, 0, 0, 0]
```

A useful sanity check: summing each row of the mask gives you back the *original*, un-padded length of that sequence — `1+1+1+0+0 = 3`, `1+1+1+1+1 = 5`, `1+1+0+0+0 = 2`. That's a fact you'll use directly in Exercise 3.

An alternative, often more efficient than padding-then-masking, is to sort sequences within a batch by length and use `pack_padded_sequence` (a PyTorch mechanism that skips computation on padded positions entirely, rather than computing on them and masking the result afterward).  code walks through both, side by side, so you can see they agree.

###  Truncation: a real tradeoff, not a technicality

Very long sequences — a full document, a long audio clip — are often **truncated** to a maximum length. This isn't just bookkeeping; it's a genuine design decision with real costs on both sides:

- **Why truncate at all**: computational cost per sequence scales with length (this resurfaces directly in Lesson 05's backpropagation-through-time cost), and very long sequences worsen certain gradient problems you'll meet in Lesson 06.
- **What it costs you**: truncating a document-classification task to its first 200 words discards everything after word 200 — if the key sentence that reveals the document's topic happens to be near the end, truncation throws it away before the network ever sees it.

There's no universally correct maximum length; it's a tradeoff you set based on your data and your compute budget, and it's worth revisiting if your model underperforms on longer examples specifically.

---

##  What "Memory" Actually Means

It's tempting to picture an RNN's hidden state as a literal buffer — a growing list that stores every past input, like a diary. **It is not that.** Recall from Lesson 01 that the hidden state update is:

$$h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h)$$

Look closely at what this equation does: at every single step, $h_t$ is a fixed-size vector (say, 128 numbers) that gets **overwritten** — not appended to — by a function of the current input and the previous hidden state. There is no growing storage. The hidden state at step 50 has exactly as many numbers in it as the hidden state at step 3.

This has an important consequence: **nothing about the architecture guarantees that any particular piece of information from step 3 will still be recoverable from the hidden state at step 50.** Whether it is or isn't depends on what the network *learns* to preserve during training — it's an empirical, learned property, not something built into the mechanism by construction. Two RNNs with the same architecture, trained differently, could differ enormously in how well they retain early information.

Why this matters going forward:

- It's exactly the reason plain ("vanilla") RNNs struggle with long-range dependencies, formalized in Lesson 06's treatment of vanishing gradients.
- It's exactly the problem LSTM and GRU (Lessons 07–08) are specifically engineered to address, by adding explicit mechanisms that make it *easier* (though still not automatic) for information to survive many steps unchanged.

A useful mental model: the hidden state is less like a diary and more like a **whiteboard that gets erased and rewritten at every step, but where the new writing is allowed to depend on what was there before.** Whether last week's notes are still legible depends entirely on whether whoever was writing chose to keep rewriting them forward.

---

##  Worked Example: From Raw Text to a Padded, Masked Batch

Let's go through the entire pipeline once, start to finish, on three tiny sentences.

**Raw data:**
```
"the cat sat"
"the dog ran fast today"
"cats sat"
```

**Step 1 — Build a vocabulary** (in practice built from your full training set; here, built just from these three sentences plus a pad token at ID 0):

```
{"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "dog": 4,
 "ran": 5, "fast": 6, "today": 7, "cats": 8}
```

**Step 2 — Tokenize and numericalize each sentence:**

```
"the cat sat"            -> ["the", "cat", "sat"]                    -> [1, 2, 3]
"the dog ran fast today" -> ["the", "dog", "ran", "fast", "today"]   -> [1, 4, 5, 6, 7]
"cats sat"                -> ["cats", "sat"]                          -> [8, 3]
```

**Step 3 — Pad to the longest sequence** (length 5):

```
[1, 2, 3, 0, 0]
[1, 4, 5, 6, 7]
[8, 3, 0, 0, 0]
```

**Step 4 — Build the mask:**

```
[1, 1, 1, 0, 0]
[1, 1, 1, 1, 1]
[1, 1, 0, 0, 0]
```

The result is a `(batch=3, T=5)` tensor of token IDs plus a matching `(3, 5)` mask — exactly the shape an RNN layer (or an embedding layer feeding into one) expects.

---

##  Use It: Code

###  From scratch: tokenizing, padding, and masking

```python
sentences = [
    "the cat sat",
    "the dog ran fast today",
    "cats sat",
]

# --- Step 1: build a vocabulary ---
vocab = {"<pad>": 0}
for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

# --- Step 2: tokenize + numericalize ---
sequences = [[vocab[word] for word in s.split()] for s in sentences]
print("Numericalized:", sequences)

# --- Step 3: pad to the longest sequence ---
max_len = max(len(s) for s in sequences)
padded = [s + [vocab["<pad>"]] * (max_len - len(s)) for s in sequences]
print("Padded:", padded)

# --- Step 4: build the mask ---
mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in sequences]
print("Mask:", mask)
```

```
Numericalized: [[1, 2, 3], [1, 4, 5, 6, 7], [8, 3]]
Padded: [[1, 2, 3, 0, 0], [1, 4, 5, 6, 7], [8, 3, 0, 0, 0]]
Mask: [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
```

This matches  exactly — good confirmation that the by-hand version and the code agree.

###  Manual masking vs. `pack_padded_sequence`

```python
import torch
import torch.nn as nn

padded_tensor = torch.tensor(padded)      # shape (3, 5)
lengths = torch.tensor([len(s) for s in sequences])  # [3, 5, 2]

embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4, padding_idx=0)
rnn = nn.RNN(input_size=4, hidden_size=6, batch_first=True)

embedded = embedding(padded_tensor)       # shape (3, 5, 4)

# Naive: run the RNN over the padded tensor with no awareness of padding.
naive_output, naive_hidden = rnn(embedded)

# Correct: pack the sequence so the RNN skips computation on padded positions.
packed = nn.utils.rnn.pack_padded_sequence(
    embedded, lengths, batch_first=True, enforce_sorted=False
)
packed_output, packed_hidden = rnn(packed)

print("Naive final hidden state (includes padding effects):\n", naive_hidden)
print("Packed final hidden state (padding ignored):\n", packed_hidden)
```

Run this yourself and compare `naive_hidden` to `packed_hidden` for the two sequences that actually contain padding (rows 0 and 2) — they'll differ, because the naive version let padding tokens influence the final hidden state, while the packed version stopped updating each sequence's hidden state the moment it hit that sequence's true length. This is the direct, hands-on version of Exercise 4 below.

---

## Exercises

**Building the pipeline**

1. Tokenize three sentences of your own choosing, of different lengths (splitting on whitespace is enough), build a small vocabulary by hand, and convert each sentence to a list of integer IDs.
2. Pad the three tokenized sentences from Exercise 1 to a common length and construct the corresponding mask tensor. Double check: does the shape of your padded batch match `(batch, T)` as defined ?
3. Given a batch of padded sequences and their mask, compute the average sequence length in the batch using **only** the mask (not the original lengths) — i.e., sum each row of the mask, then average those sums. Confirm your answer matches the true average length you already know from Exercise 1.

**Truncation and tradeoffs**

4. Suppose you're building a spam classifier for emails, and you decide to truncate every email to its first 50 words to save compute. Describe a realistic scenario where this truncation choice would cause the classifier to make a mistake it wouldn't otherwise make.

---

## Key Terms

| Term | What it actually means |
|---|---|
| Tokenization | Splitting raw text into discrete units (words, subwords, or characters) before converting them to numeric IDs |
| Numericalization | Mapping each token to an integer ID using a fixed vocabulary |
| Padding | Extending shorter sequences in a batch with a filler value (often ID 0) so every sequence in the batch has the same length |
| Mask | A same-shaped tensor of 1s and 0s marking which positions are real data versus padding, used to make downstream computations ignore padded positions |
| Truncation | Cutting a sequence down to a maximum length, discarding everything past that point — a real tradeoff between compute cost and information loss, not a mere implementation detail |
| pack_padded_sequence | A PyTorch utility that lets an RNN skip computation on padded positions directly, rather than computing on them and masking afterward |
| Hidden state as memory | A fixed-size vector overwritten (not appended to) at every step; whether early information survives many steps is a learned, empirical property of training, not a structural guarantee |