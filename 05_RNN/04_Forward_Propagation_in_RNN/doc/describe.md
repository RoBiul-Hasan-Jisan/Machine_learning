# 04. Forward Propagation in RNN

## Learning Objectives

- Chain the single-step RNN cell (Lesson 03) into a full forward pass over an entire sequence
- Implement the three input/output patterns from Lesson 01 (many-to-one, many-to-many, one-to-many) concretely
- Verify a from-scratch sequence forward pass against `torch.nn.RNN`

## The Problem

Lesson 03 defined what happens at *one* time step. Forward propagation is running that single-step computation repeatedly across an entire sequence, feeding each step's hidden state into the next, and reading out outputs at whichever steps the task requires — the mechanics behind the "same RNN cell reused at every time step" picture from Lesson 01.

## The Concept

### Unrolling the RNN cell across a sequence

```
h_0 = zeros

for t in 1..T:
    h_t = tanh(W_xh @ x_t + W_hh @ h_(t-1) + b_h)     <- Lesson 03's cell equation
    y_t = W_hy @ h_t + b_y                              <- optional, only if this step needs an output

return all h_t (and/or all y_t), and the final h_T
```

This is called "unrolling" the RNN: conceptually, applying the same cell `T` times produces a computation graph that looks like `T` copies of the same small network, chained together — which is exactly the picture from Lesson 01's diagram, now backed by the concrete equation. This unrolled view matters directly for Lesson 05 (backpropagation through time), since gradients flow backward through every one of these `T` copies.

### Many-to-one: read the output only at the last step

For tasks like sentiment classification (a whole sequence maps to one label), only the final hidden state matters for the prediction — earlier hidden states exist only to build up to it:

```
x_1 -> h_1 -> x_2 -> h_2 -> ... -> x_T -> h_T -> y  (single output, read from h_T only)
```

```python
def rnn_forward_many_to_one(sequence, W_xh, W_hh, b_h, W_hy, b_y):
    h = np.zeros(W_hh.shape[0])
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
    y = W_hy @ h + b_y     # only computed once, after the loop
    return y
```

### Many-to-many: read an output at every step

For tasks like part-of-speech tagging (every word gets its own label) or language modeling (predict the next token at every position), an output is read out at every time step:

```
x_1 -> h_1 -> y_1
x_2 -> h_2 -> y_2
...
x_T -> h_T -> y_T
```

```python
def rnn_forward_many_to_many(sequence, W_xh, W_hh, b_h, W_hy, b_y):
    h = np.zeros(W_hh.shape[0])
    outputs = []
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        y_t = W_hy @ h + b_y
        outputs.append(y_t)
    return outputs, h
```

There's a second many-to-many variant — encoder-decoder style, where the entire input is consumed first and outputs are only produced afterward, at a *different* number of steps than the input had (e.g. translating a 5-word sentence into a 7-word sentence) — covered specifically in Lesson 11 (Sequence-to-Sequence Models), since it needs an additional idea (two separate RNNs) beyond what a single unrolled RNN provides here.

### One-to-many: a single input drives an entire generated sequence

For tasks like image captioning (one image produces a whole sentence), a single input is used to initialize the hidden state (or is fed in only at the first step), and the network then generates a sequence of outputs, typically feeding each step's own output back in as the next step's input:

```
x -> h_1 -> y_1 -> h_2 -> y_2 -> h_3 -> y_3 -> ...
        (y_1 fed back in as input to produce h_2, and so on)
```

This pattern — feeding a step's own output back in as the next input — reappears centrally in Lesson 14 (Text Generation), where it's exactly how a trained language model produces text one token at a time.

### Batched forward propagation

In practice, forward propagation runs on a batch of sequences at once (Lesson 02's padded/masked batches), not one sequence at a time. Every operation extends with a batch dimension: `x_t` becomes shape `(batch, input_size)`, `h_t` becomes `(batch, hidden_size)`, and the same weight matrices are applied identically across every sequence in the batch, in parallel, at each time step — this is what `torch.nn.RNN` does internally when given a `(batch, T, input_size)` tensor.

See `code/forward_prop_rnn_demo.py` for from-scratch implementations of the many-to-one and many-to-many patterns, verified against `torch.nn.RNN` given identical weights, plus a batched version.

## Exercises

1. Implement `rnn_forward_many_to_one` from scratch and verify its final hidden state matches `torch.nn.RNN`'s final hidden state output, given the same weights and the same input sequence.
2. Implement `rnn_forward_many_to_many` and verify every step's hidden state matches the corresponding position in `torch.nn.RNN`'s full output sequence (not just the final one).
3. Modify the many-to-many implementation to only compute `y_t` for every *third* time step (a made-up but illustrative "sparse many-to-many" pattern), and explain what real-world task structure this could correspond to.
4. Batch three sequences of different lengths together (using Lesson 02's padding), run them through a batched forward pass, and confirm the outputs for the un-padded positions match what you'd get running each sequence individually.

## Key Terms

| Term | What it actually means |
|---|---|
| Unrolling | Viewing an RNN's repeated application of the same cell across time steps as a chain of copies, useful for reasoning about both forward and backward computation |
| Many-to-one | An RNN usage pattern where a whole input sequence produces a single output, read from the final hidden state |
| Many-to-many | An RNN usage pattern where an output is produced at every time step (or, in the encoder-decoder variant, after the full input is consumed) |
| One-to-many | An RNN usage pattern where a single input drives the generation of an entire output sequence, often by feeding each output back in as the next input |
