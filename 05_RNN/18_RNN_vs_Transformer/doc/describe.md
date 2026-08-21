# 18. RNN vs Transformer

## Learning Objectives

- Explain why RNNs are fundamentally hard to parallelize during training, and why Transformers aren't
- Compare RNN attention (Lesson 12) to Transformer self-attention, and identify what generalized between them
- Identify the situations where RNNs remain a reasonable or preferable choice today

## The Problem

Lesson 12 introduced attention as a fix for the encoder-decoder bottleneck in sequence-to-sequence RNNs (Lesson 11): instead of compressing an entire input into one fixed-size context vector, let the decoder look back at *all* encoder hidden states and weight them by relevance. The Transformer architecture (Vaswani et al., 2017, "Attention Is All You Need") asked a more radical question: if attention is doing most of the useful work, do we need the recurrence at all? The answer — remove the RNN entirely and build a model from attention alone — turned out to scale dramatically better and now underlies essentially all large-scale language models.

## The Concept

### The core limitation attention-only architectures solve: sequential computation

Every RNN variant in this module — plain, LSTM, GRU, bidirectional, stacked — shares one structural property: step `t`'s computation depends on step `t-1`'s completed result (Lesson 03's cell equation, `h_t = f(x_t, h_{t-1})`). This is precisely what gives RNNs their sequential, memory-carrying behavior, but it also means step `t` **cannot start until step `t-1` finishes** — a sequence of length 1000 requires 1000 sequential computation steps, no matter how much parallel hardware (GPUs, with thousands of parallel cores) is available. This is a hard architectural constraint, not a tuning choice, and it directly limits how fast RNNs can be trained on long sequences or large datasets, regardless of hardware.

A Transformer removes this constraint by dropping recurrence entirely. Every position in a Transformer's self-attention layer is computed using only the *input* sequence (not a chain of previous outputs), so every position's computation can run **in parallel**, fully exploiting modern GPU/TPU hardware in a way an RNN's step-by-step dependency structure cannot.

```
RNN:          h_1 -> h_2 -> h_3 -> ... -> h_T     (strictly sequential; step t needs step t-1's result)

Transformer:  all positions' self-attention computed simultaneously, using only the input
              (no step depends on completing a previous step's OUTPUT first)
```

### From RNN attention to self-attention

Lesson 12's attention mechanism computed, for each decoder step, a weighted combination of *encoder* hidden states, using a compatibility score between the decoder's current state (the query) and each encoder state (the keys), with the encoder states themselves also serving as the values being combined:

```
Lesson 12's attention:  query = decoder's current hidden state
                         keys/values = encoder's hidden states (from an RNN)
                         context = weighted sum of values, weights from query-key compatibility
```

Transformer **self-attention** generalizes this same query/key/value mechanism, but applies it *within* a single sequence rather than between an encoder and decoder — every position attends to every other position in the *same* sequence, using learned linear projections of the input (not RNN hidden states) to produce queries, keys, and values:

```
Self-attention:  Q = X @ W_q,  K = X @ W_k,  V = X @ W_v      (X = input sequence, all positions at once)

attention_weights = softmax(Q @ K^T / sqrt(d_k))               (every position's compatibility with every other)
output = attention_weights @ V                                  (weighted combination of values)
```

Critically, `Q`, `K`, and `V` here are all computed directly from the input `X` via matrix multiplications — operations that are trivially parallelizable across every position at once, unlike an RNN hidden state that must be computed one step after another. This is the mechanism that inherits attention's core benefit (every position can directly access information from every other position, regardless of distance — solving Lesson 06's long-range dependency problem in a completely different way than LSTM/GRU's gating) while discarding the sequential bottleneck that made RNN training slow to parallelize.

### What Transformers need instead of recurrence

Removing recurrence means removing the one thing that gave an RNN any sense of sequence order — a plain self-attention layer, on its own, treats the input as an unordered set of positions. Transformers add **positional encodings** (a fixed or learned vector added to each position's input, encoding "this is position 5" numerically) specifically to reintroduce order information, since nothing else in the architecture would otherwise distinguish "the cat sat" from "sat the cat."

### Practical comparison

| Factor | RNN (LSTM/GRU) | Transformer |
|---|---|---|
| Parallelizable across sequence positions during training | No — strictly sequential | Yes — all positions computed simultaneously |
| Handles long-range dependencies | Struggles even with LSTM/GRU gating on very long sequences | Direct access between any two positions via attention, regardless of distance |
| Memory cost | Roughly linear in sequence length | Quadratic in sequence length (every position attends to every other) |
| Data/compute needed to train well | Lower — the recurrent structure is a useful built-in bias for sequential data | Higher — needs to learn sequential structure largely from data, via positional encodings |
| Streaming / online inference | Natural fit — one step at a time, low latency | Requires adaptation (e.g. caching, or specialized streaming variants) |
| Current dominance for large-scale language tasks | Largely superseded | The standard architecture for modern LLMs |

The quadratic memory cost row matters in practice: a Transformer's self-attention computes compatibility between every pair of positions, so memory scales with `T²` for a sequence of length `T`, whereas an RNN's memory cost scales linearly with `T`. This is why very long sequences (extremely long documents, high-resolution audio) still sometimes favor RNN-like or hybrid approaches, or specialized attention variants designed specifically to reduce this quadratic cost.

### When RNNs remain a reasonable choice

Despite Transformers' dominance for large-scale text tasks, RNNs (and LSTM/GRU specifically) remain genuinely useful in several situations, directly following from the comparison above:

- **Streaming or real-time tasks** where predictions must be produced incrementally, one step at a time, with low latency — an RNN's step-by-step structure is a natural fit; a standard Transformer is not (Lesson 09's bidirectionality discussion made a related point about online tasks).
- **Smaller-scale or resource-constrained settings**, where an RNN's linear memory scaling and lower data requirements can be a genuine practical advantage over a Transformer's quadratic memory cost and larger data appetite.
- **Time series forecasting** (Lesson 15) on moderate-length sequences, where RNNs remain a strong, simple, well-understood default, though Transformer-based and hybrid approaches are increasingly competitive here too.
- **As a conceptual foundation.** Understanding attention as it was originally introduced — fixing a specific, concrete bottleneck in an RNN encoder-decoder (Lesson 11's fixed-size context vector problem) — is what makes the Transformer's design choices legible as *solutions to real problems*, rather than an arbitrary architecture that happened to work well.

See `code/rnn_vs_transformer_demo.py` for a self-attention layer implemented from scratch (matching the pattern above, using PyTorch for the underlying tensor operations), a runtime comparison showing an RNN's sequential processing cost vs a Transformer-style layer's parallel cost as sequence length grows, and a minimal positional encoding implementation.

## Exercises

1. Implement self-attention from scratch (`Q = X @ W_q`, etc., following the equations above) and verify your output shape and values against `torch.nn.MultiheadAttention` on a small input, using matching weights.
2. Time (using `torch.utils.benchmark` or simple wall-clock timing) an `nn.LSTM` versus a self-attention layer processing sequences of length 50, 200, and 800. Observe how the relative speed gap changes as sequence length grows.
3. Implement a sinusoidal positional encoding (as used in the original Transformer paper) and add it to a random input embedding sequence. Visualize a few dimensions of the encoding across positions to see the pattern.
4. For each of the following tasks, decide whether an RNN or a Transformer is the more natural fit, and justify your answer using this lesson's comparison table: (a) live captioning of a video call, (b) translating a static, already-written document, (c) forecasting tomorrow's temperature from the last 30 days of readings, (d) answering questions about a 500-page document.

## Key Terms

| Term | What it actually means |
|---|---|
| Transformer | An architecture built entirely from attention mechanisms (no recurrence), enabling full parallelization across sequence positions during training |
| Self-attention | Attention applied within a single sequence, where every position computes a weighted combination of every other position's values, using learned query/key/value projections |
| Positional encoding | A vector added to each position's input in a Transformer to encode sequence order, since self-attention alone has no inherent notion of position |
| Parallelizability | The degree to which a computation can be split across multiple processors simultaneously; the key structural advantage Transformers have over RNNs |
| Quadratic memory cost | The property that standard self-attention's memory usage scales with the square of sequence length, since every position attends to every other position |
