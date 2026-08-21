# 05. Backpropagation Through Time

## Learning Objectives

- Explain how the chain rule extends to a network unrolled across time steps
- Derive why a shared weight's total gradient is a sum over every time step it was used in
- Implement backpropagation through time (BPTT) for a small RNN and verify it against PyTorch's autograd

## The Problem

Lesson 04's forward pass reuses the exact same weights (`W_xh`, `W_hh`, `b_h`) at every time step. Training the network means computing the gradient of the loss with respect to those weights — but each weight influenced the loss through *every* time step it participated in, not just one. Backpropagation Through Time (BPTT) is the specific application of the chain rule that correctly accounts for this repeated use of the same weights across the unrolled sequence.

## The Concept

### Unrolling makes BPTT look like ordinary backprop, with one twist

Lesson 04's "unrolling" picture — treating the RNN as `T` chained copies of the same cell — makes forward propagation easy to reason about. Backpropagation through the unrolled graph works exactly like backpropagation through any deep feedforward network (covered generally in the CNN module's Lesson 09): gradients flow backward from the loss, through each time step, via the chain rule. The twist: since the *same* `W_hh` (and `W_xh`, `b_h`) was used at every one of the `T` steps, the total gradient for `W_hh` is the **sum** of its contribution at every individual time step, not just the contribution from the last one.

```
Loss depends on h_T, which depends on h_(T-1) (and W_hh), which depends on h_(T-2) (and W_hh), ...

dLoss/dW_hh = sum over all time steps t:  (gradient of loss w.r.t. h_t) * (dh_t/dW_hh, holding
                                            earlier h's fixed -- i.e., W_hh's DIRECT contribution
                                            at that specific step)
```

This is the same principle as a convolutional filter's gradient in the CNN module (Lesson 09 there): a shared weight's total gradient sums its contribution across every place it was used — spatial positions for a conv filter, time steps for an RNN's recurrent weights.

### The gradient flowing backward through the hidden state

At each step going backward, the gradient with respect to `h_t` has two sources: the direct loss contribution at that step (if `y_t` was read out and contributes to the loss, per Lesson 04's many-to-many pattern), and the gradient flowing back from `h_{t+1}`, since `h_t` was used to compute `h_{t+1}`:

```
dLoss/dh_t = dLoss_t/dh_t   (direct contribution, if y_t exists)
           + dLoss/dh_(t+1) * dh_(t+1)/dh_t     (contribution via the next step)
```

Working this out at every step, from `t=T` backward to `t=1`, and accumulating each step's contribution to `dLoss/dW_hh`, `dLoss/dW_xh`, and `dLoss/db_h` along the way, is exactly what BPTT does. In practice you never derive or implement this by hand in real projects — `loss.backward()` in PyTorch handles it automatically, walking the unrolled computation graph exactly as described — but understanding the mechanism concretely is what makes the gradient problems in Lesson 06 (which arise directly from this backward chain through many time steps) make sense rather than feeling like an unexplained fact about RNNs.

### Why this is expensive for long sequences

BPTT's cost — both compute and memory — scales with sequence length `T`, since every one of the `T` unrolled steps needs its intermediate values cached (Lesson 08 of the CNN module's "cache" concept, here across time instead of across layers) for the backward pass to use. This is the direct reason very long sequences are often truncated (Lesson 02) or processed with **truncated BPTT** — splitting a long sequence into shorter chunks, running full forward/backward passes within each chunk, and carrying the hidden state (but not the full gradient history) across chunk boundaries. Truncated BPTT trades some gradient accuracy (the network can no longer learn dependencies that span across a chunk boundary) for tractable memory and compute on long sequences.

### Implementing BPTT from scratch

```python
def rnn_backward_step(dh_t, h_t, h_prev, x_t, W_hh):
    """One step of BPTT: given the gradient flowing INTO h_t from the future,
    compute the gradient contributions at this step and the gradient to pass
    further backward (into h_(t-1))."""
    dz = dh_t * (1 - h_t ** 2)          # tanh'(z) = 1 - tanh(z)^2 = 1 - h_t^2
    dW_xh = np.outer(dz, x_t)
    dW_hh = np.outer(dz, h_prev)
    db_h = dz
    dh_prev = W_hh.T @ dz               # gradient to pass to the PREVIOUS time step
    return dW_xh, dW_hh, db_h, dh_prev
```

Called once per time step, working backward from `t=T` to `t=1`, accumulating `dW_xh`, `dW_hh`, `db_h` across every call (since the same weights are shared across steps, per the sum above) and threading `dh_prev` from one call into the next call's `dh_t` — this loop, run over the full unrolled sequence, is BPTT in its entirety.

See `code/bptt_demo.py` for the complete from-scratch BPTT implementation over a multi-step sequence, with every accumulated gradient (`dW_xh`, `dW_hh`, `db_h`) verified against PyTorch's autograd on an equivalent computation.

## Exercises

1. Implement `rnn_backward_step` and manually run it backward over a 4-step sequence, accumulating `dW_hh` across all 4 steps. Verify the accumulated total against `torch.autograd`'s `.grad` on `W_hh` for the same forward computation.
2. For the same 4-step sequence, isolate and print just the *contribution* to `dW_hh` from each individual time step (before summing). Confirm they're generally different from each other, illustrating that "the same weight, different time steps" genuinely contributes different amounts to the total gradient.
3. Implement a simple version of truncated BPTT: split an 8-step sequence into two 4-step chunks, carry only the hidden state (not gradient history) across the chunk boundary, and compare the resulting `dW_hh` to the gradient from full (untruncated) BPTT over all 8 steps.
4. Time how BPTT's forward+backward cost scales as you increase sequence length from 10 to 100 to 1000 steps, using `torch.nn.RNN` and `loss.backward()`, and relate the scaling to the caching requirement described above.

## Key Terms

| Term | What it actually means |
|---|---|
| Backpropagation Through Time (BPTT) | Backpropagation applied to an RNN's unrolled computation graph, where a shared weight's total gradient sums its contribution across every time step it was used in |
| Truncated BPTT | Splitting a long sequence into shorter chunks for backpropagation, carrying the hidden state but not the full gradient history across chunk boundaries |
| Unrolled computation graph | The view of an RNN's forward pass as a chain of identical cell computations across time steps, used to reason about both forward and backward passes |
