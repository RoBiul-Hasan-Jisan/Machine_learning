# 08. GRU

## Learning Objectives

- Write the equations for a GRU's reset and update gates and its hidden state update
- Explain how GRU achieves LSTM-like gradient benefits with a simpler structure (no separate cell state)
- Compare LSTM and GRU in parameter count, computational cost, and typical empirical performance

## The Problem

LSTM (Lesson 07) solves the vanishing gradient problem with three gates and a separate cell state — effective, but more parameters and more computation per step than a plain RNN. The Gated Recurrent Unit (GRU), introduced by Cho et al. in 2014, asks whether a similar benefit — gated, additive updates that let information persist across steps — can be achieved with a simpler structure: two gates instead of three, and no separate cell state.

## The Concept

### GRU's gates

```
Update gate:  z_t = sigmoid(W_z @ [h_(t-1), x_t] + b_z)
Reset gate:   r_t = sigmoid(W_r @ [h_(t-1), x_t] + b_r)
```

Two gates instead of LSTM's three. The **update gate** plays a role similar to LSTM's combined forget-and-input gates: it controls the balance between keeping the old hidden state and incorporating new information. The **reset gate** controls how much of the previous hidden state gets used when computing the new candidate.

### The candidate hidden state and the update

```
Candidate:   h_tilde_t = tanh(W_h @ [r_t * h_(t-1), x_t] + b_h)

Update:      h_t = (1 - z_t) * h_(t-1) + z_t * h_tilde_t
                     ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
                     keep old            add new candidate
```

Notice the update equation's structure: `z_t` alone controls the entire keep-vs-replace tradeoff (`1 - z_t` and `z_t` are complementary — they always sum to 1), unlike LSTM's forget gate `f_t` and input gate `i_t`, which are computed independently and don't need to sum to anything in particular. This is GRU's central simplification: one gate doing the job LSTM splits across two, at the cost of losing that independence (a GRU can't, for example, simultaneously "forget a lot of the old state" and "add only a little new information" — the two are coupled through `z_t`).

The reset gate `r_t` gates the previous hidden state *before* it's used to compute the candidate — when `r_t` is close to 0, the candidate `h_tilde_t` is computed almost entirely from the current input, effectively ignoring the past when proposing new content, even though the old hidden state may still be substantially preserved via `1 - z_t` in the final update.

### No separate cell state

This is GRU's other major simplification relative to LSTM: there's only `h_t`, serving both the role LSTM's `c_t` played (a channel that can persist across steps via the gated update) and the role LSTM's `h_t` played (used directly for output and passed to the next layer). Fewer distinct pieces of state, fewer weight matrices, and — since the update equation `h_t = (1-z_t)*h_(t-1) + z_t*h_tilde_t` is structurally identical in spirit to LSTM's `c_t = f_t*c_(t-1) + i_t*c_tilde_t` — GRU gets a very similar gradient-flow benefit from Lesson 07's analysis: when `z_t` stays small, `h_t ≈ h_(t-1)`, letting gradients flow backward largely undiminished through many steps, addressing Lesson 06's vanishing gradient problem by the same additive-update mechanism LSTM uses.

### GRU vs LSTM: fewer parameters, comparable performance

```
LSTM: 3 gates + 1 candidate = 4 weight matrices (each combining h and x)
GRU:  2 gates + 1 candidate = 3 weight matrices

For hidden_size H and input_size D:
  LSTM parameters (weights only): 4 * H * (H + D)
  GRU parameters (weights only):  3 * H * (H + D)

GRU has 25% fewer parameters than an LSTM of the same hidden size.
```

Empirically, across many tasks, GRU and LSTM perform comparably — neither dominates the other universally, and the choice often comes down to GRU's lower parameter count and slightly faster training/inference (fewer matrix multiplications per step) versus LSTM's separate cell state occasionally giving it an edge on tasks demanding more complex, longer-range memory management. In practice, both are reasonable defaults, and the right choice for a specific project is often best settled empirically (try both, compare validation performance) rather than by a universal rule — this is one of the few genuinely "it depends, try both" situations in this module.

### Implementing a GRU cell from scratch

```python
def gru_cell_forward(x_t, h_prev, W_z, b_z, W_r, b_r, W_h, b_h):
    combined = np.concatenate([h_prev, x_t])

    z_t = sigmoid(W_z @ combined + b_z)
    r_t = sigmoid(W_r @ combined + b_r)

    combined_reset = np.concatenate([r_t * h_prev, x_t])
    h_tilde_t = np.tanh(W_h @ combined_reset + b_h)

    h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
    return h_t
```

Note: this is the standard textbook formula (following Cho et al.'s original paper), with input and hidden weights combined into one matrix per gate for readability. Production implementations differ in two small but real ways worth knowing about if you ever compare against them directly: PyTorch's `nn.GRUCell` keeps input-facing and hidden-facing weights and biases as separate matrices and applies the reset gate specifically to the hidden branch — `r_t * (W_h_hidden @ h_prev + b_h_hidden)` — rather than to a pre-concatenated vector; and PyTorch's update-gate convention is flipped relative to the formula above, with `z_t` close to 1 meaning "keep the old hidden state" rather than "take the new candidate." Both are legitimate GRUs — only the labeling differs — but the flip means naively copying the formula above into code compared against PyTorch will silently disagree unless you either flip `z_t`'s role or use `1 - z_t` in the right place. The code demo implements PyTorch's exact version for a bit-for-bit numerical match.

See `code/gru_demo.py` for the complete implementation verified against `torch.nn.GRUCell`, a side-by-side parameter count comparison against LSTM at matching hidden sizes, and a small training speed comparison.

## Exercises

1. Implement `gru_cell_forward` and verify it matches `torch.nn.GRUCell` exactly, given the same weights. Note PyTorch keeps input/hidden weights and biases separate per gate, and gates the *hidden bias* specifically with the reset gate — check its documentation for the exact formula before comparing.
2. Compute the exact parameter count for an LSTM and a GRU at `hidden_size=128, input_size=64`, and confirm the ratio matches the `4:3` relationship derived above.
3. Train an LSTM and a GRU of matching hidden size on the same small sequence task (e.g. the long-range dependency task from Lesson 06's exercises) for the same number of epochs, and compare final accuracy and wall-clock training time.
4. Set `r_t` to a fixed value of 0 for every step (instead of letting it be learned) and observe how the candidate hidden state computation changes. Explain in words what this ablation removes from the network's capability.

## Key Terms

| Term | What it actually means |
|---|---|
| GRU (Gated Recurrent Unit) | An RNN variant using two gates (update, reset) and no separate cell state, achieving similar gradient-flow benefits to LSTM with fewer parameters |
| Update gate | The GRU gate controlling the balance between keeping the previous hidden state and incorporating the new candidate |
| Reset gate | The GRU gate controlling how much of the previous hidden state is used when computing the new candidate hidden state |
| Candidate hidden state | The new content proposed at the current step, blended with the previous hidden state according to the update gate |
