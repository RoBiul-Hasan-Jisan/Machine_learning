# 07. LSTM

## Learning Objectives

- Explain the cell state and why it provides a more direct path for gradients than a plain RNN's hidden state
- Write the equations for all three LSTM gates and the cell state update
- Implement an LSTM cell from scratch and verify it against `torch.nn.LSTMCell`

## The Problem

Lesson 06 diagnosed the core issue with plain RNNs: the hidden state is fully overwritten at every step by a `tanh` of a linear combination, and the repeated multiplication this creates during backpropagation causes gradients (and therefore the ability to learn long-range dependencies) to vanish over long sequences. Long Short-Term Memory (LSTM), introduced by Hochreiter & Schmidhuber in 1997, fixes this with a structural change: give the network a *second* piece of state — the cell state — specifically designed to let information flow across many time steps with only small, deliberate modifications, rather than being overwritten wholesale every step.

## The Concept

### Two pieces of state instead of one

A plain RNN has one hidden state, fully recomputed every step (Lesson 03). LSTM maintains two:

```
h_t: the hidden state (same role as before -- used for output, passed to next layer)
c_t: the cell state (NEW -- a more protected channel for long-term information)
```

The cell state is the key innovation: rather than being replaced at every step, it's updated through addition and elementwise multiplication by values between 0 and 1 (gates), which — critically for the vanishing gradient problem — allows gradients to flow backward through many steps via addition rather than repeated matrix multiplication, avoiding the shrink-or-explode dynamic from Lesson 06.

### Gates: learned, per-step control over what to remember and forget

An LSTM cell has three gates, each a small neural network layer (a linear transformation followed by a sigmoid, producing values between 0 and 1) that controls how information flows into, out of, and through the cell state:

```
Forget gate:  f_t = sigmoid(W_f @ [h_(t-1), x_t] + b_f)     "how much of the old cell state to keep"
Input gate:   i_t = sigmoid(W_i @ [h_(t-1), x_t] + b_i)     "how much of the new candidate to add"
Output gate:  o_t = sigmoid(W_o @ [h_(t-1), x_t] + b_o)     "how much of the cell state to expose as h_t"
```

`[h_(t-1), x_t]` denotes concatenating the previous hidden state and the current input into one vector before the linear transformation — a common notational shorthand; in practice this is equivalent to (and usually implemented as) two separate weight matrices, one for `h_(t-1)` and one for `x_t`, added together, mirroring the `W_hh` / `W_xh` split from Lesson 03. Sigmoid squashes each gate's output to `[0, 1]`, which is what makes it interpretable as "how much of this passes through" — 0 means "block completely," 1 means "pass through entirely."

### The candidate cell state and the cell state update

```
Candidate:       c_tilde_t = tanh(W_c @ [h_(t-1), x_t] + b_c)     "new information proposed at this step"

Cell state update:   c_t = f_t * c_(t-1) + i_t * c_tilde_t
                            ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
                            keep old        add new
                            (gated)         (gated)
```

This is the central equation. The new cell state is a gated blend of the old cell state and a new candidate — `*` here is elementwise multiplication, not matrix multiplication. If the forget gate `f_t` is close to 1 and the input gate `i_t` is close to 0, `c_t ≈ c_(t-1)`: the cell state passes through almost unchanged. This is exactly the mechanism that lets relevant information from early in a sequence survive, largely untouched, across many later steps — something a plain RNN's `tanh`-squashed hidden state has no direct way to do.

### The hidden state output

```
h_t = o_t * tanh(c_t)
```

The hidden state is a gated, `tanh`-squashed *view* of the cell state — what the cell "chooses to expose" at this step, used for any output computation (Lesson 04's `y_t = W_hy @ h_t + b_y`) and passed to the next step (and, in a stacked architecture, to the next layer, covered in Lesson 10).

### Why this addresses vanishing gradients

Backpropagating through `c_t = f_t * c_(t-1) + i_t * c_tilde_t`, the gradient path back to `c_(t-1)` goes through `f_t` — an elementwise multiplication by a learned gate value, not a repeated matrix multiplication by a fixed `W_hh` at every step (Lesson 06's core problem). When the network learns to keep `f_t` close to 1 for information that needs to persist, gradients can flow backward through many steps largely undiminished — the network has an explicit, *learned* mechanism for preserving long-range information, rather than relying on `tanh`'s derivative happening to stay large by chance. This doesn't make LSTMs immune to vanishing/exploding gradients entirely (gradient clipping, Lesson 06, is still standard practice), but it addresses the structural cause far more directly than any training trick could.

### Implementing an LSTM cell from scratch

```python
def lstm_cell_forward(x_t, h_prev, c_prev, params):
    combined = np.concatenate([h_prev, x_t])

    f_t = sigmoid(params["W_f"] @ combined + params["b_f"])
    i_t = sigmoid(params["W_i"] @ combined + params["b_i"])
    o_t = sigmoid(params["W_o"] @ combined + params["b_o"])
    c_tilde_t = np.tanh(params["W_c"] @ combined + params["b_c"])

    c_t = f_t * c_prev + i_t * c_tilde_t
    h_t = o_t * np.tanh(c_t)

    return h_t, c_t
```

See `code/lstm_demo.py` for the complete implementation verified against `torch.nn.LSTMCell`, plus a direct empirical comparison of gradient magnitude over a long sequence between a plain RNN and an LSTM, showing the practical benefit from Lesson 06's diagnosis.

## Exercises

1. Implement `lstm_cell_forward` and verify it matches `torch.nn.LSTMCell` exactly, given the same weights (note: PyTorch's `LSTMCell` concatenates the four gates' weights into single matrices for efficiency — check its documentation for the exact weight layout when copying weights across).
2. Set the forget gate's bias to a large positive value (e.g. 5) before training and explain, using the sigmoid function, why this is a common practical initialization trick (hint: consider what `sigmoid(5)` evaluates to and what that means for how much of `c_(t-1)` survives at the very start of training).
3. Reproduce the gradient-magnitude-vs-sequence-length comparison from Lesson 06, this time comparing a plain RNN against an LSTM, and confirm the LSTM's gradient stays substantially larger at long sequence lengths.
4. Trace through the cell state update equation by hand for 3 time steps with a toy 1-dimensional cell state, choosing `f_t` and `i_t` values that illustrate "mostly remember" vs "mostly forget and replace."

## Key Terms

| Term | What it actually means |
|---|---|
| LSTM (Long Short-Term Memory) | An RNN variant with an explicit cell state and gating mechanism, designed to preserve long-range information and address vanishing gradients |
| Cell state | LSTM's protected channel for long-term information, updated through gated addition rather than full overwriting at every step |
| Forget gate | The gate controlling how much of the previous cell state is retained at the current step |
| Input gate | The gate controlling how much new candidate information is added to the cell state at the current step |
| Output gate | The gate controlling how much of the cell state is exposed as the hidden state at the current step |
| Candidate cell state | The new information proposed at the current step, before being gated into the cell state by the input gate |
