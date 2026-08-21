# 06. Vanishing and Exploding Gradients

## Learning Objectives

- Explain why BPTT's repeated multiplication by `W_hh` causes gradients to shrink or grow exponentially with sequence length
- Demonstrate vanishing and exploding gradients empirically on a plain RNN
- Apply gradient clipping as a direct (partial) fix, and understand why it addresses exploding but not vanishing gradients

## The Problem

Lesson 05 showed the gradient flowing backward through an RNN passes through a `W_hh.T @ dz` multiplication at every single time step. For a sequence of length `T`, the gradient reaching the earliest time steps has effectively been multiplied by something like `W_hh` (and the local derivative of `tanh`) `T` times in a row. Repeated multiplication is exactly the kind of operation that either shrinks toward zero or grows without bound, depending on the scale of what's being multiplied — and for RNNs, this isn't a rare edge case, it's close to the default behavior for anything beyond short sequences.

## The Concept

### Why repeated multiplication is the core issue

Simplify the backward recurrence from Lesson 05 to isolate the repeated factor:

```
dh_t/dh_(t-1) ≈ W_hh^T * diag(tanh'(z_t))

Gradient reaching h_1 from h_T (T steps back):

dLoss/dh_1 ≈ dLoss/dh_T * PRODUCT over t=2..T of [ W_hh^T * diag(tanh'(z_t)) ]
```

That product of `T-1` matrices is the crux of the problem. Two things about it matter:

- `tanh'(z) = 1 - tanh(z)^2` is at most 1, and typically much smaller than 1 whenever `z` is far from 0 (tanh saturates for large |z|, exactly the CNN module's activation-function discussion of saturating nonlinearities). Multiplying by something `≤ 1` repeatedly, `T-1` times, tends to shrink the gradient toward zero — **vanishing gradients**.
- If `W_hh`'s largest eigenvalue happens to be greater than 1, repeated multiplication by `W_hh` can instead grow the gradient — potentially very large — as `T` increases. This is **exploding gradients**.

Both are consequences of the exact same repeated-multiplication structure; which one dominates depends on the specific scale of `W_hh` and how saturated the `tanh` activations are, and in practice a poorly-behaved RNN can show signs of both at different points in training.

### Vanishing gradients: the network can't learn long-range dependencies

If the gradient reaching an early time step is effectively zero, weight updates driven by that gradient are also effectively zero — the network cannot learn to use information from early in a long sequence to affect later predictions, no matter how relevant that information actually is. Concretely: a plain RNN trying to learn that "the subject introduced in word 2 determines the correct verb conjugation in word 40" will struggle, because the gradient connecting word 40's error back to word 2's contribution has vanished by the time it gets there. This is the single biggest practical limitation of plain RNNs, and precisely the problem LSTM and GRU (Lessons 07-08) are architected to solve.

### Exploding gradients: training becomes unstable

An exploding gradient produces a huge weight update in one step, which can throw the network's weights into a wildly different (and usually much worse) region, sometimes producing `NaN` values (as seen in the CNN module's training troubleshooting table) and derailing training entirely. Unlike vanishing gradients, exploding gradients are comparatively easy to detect (loss suddenly spikes or becomes `NaN`) and have a direct, effective fix.

### Gradient clipping: a direct fix for exploding gradients

Before applying the optimizer's update, rescale the gradient if its norm exceeds a threshold, capping how large any single update step can be:

```
if ||gradient|| > threshold:
    gradient = gradient * (threshold / ||gradient||)
```

This preserves the gradient's *direction* (still a genuinely useful update signal) while capping its *magnitude*, preventing any single step from being catastrophically large. It's cheap, simple, and standard practice for any RNN training loop:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

Gradient clipping fixes exploding gradients directly, but does **not** address vanishing gradients — clipping only caps large gradients; it can't manufacture a useful gradient signal where the true gradient has genuinely shrunk to near zero. This is why gradient clipping is standard practice *alongside* architectural fixes (Lessons 07-08), not a substitute for them.

### The real fix for vanishing gradients: architecture, not just training tricks

Gradient clipping, careful weight initialization, and using a less-saturating activation don't solve the fundamental issue that a plain RNN's hidden state is *overwritten* at every step (Lesson 02's point about the hidden state not being a literal buffer). LSTM (Lesson 07) and GRU (Lesson 08) instead give the network an explicit, learnable mechanism to *preserve* information across many time steps largely unchanged when needed — directly targeting the repeated-multiplication problem at its source, rather than working around its symptoms.

See `code/vanishing_exploding_demo.py` for an empirical demonstration: tracking gradient magnitude at the earliest time step as sequence length increases, for both a "shrinking" weight configuration (vanishing) and a "growing" weight configuration (exploding), plus a demonstration of gradient clipping in action.

## Exercises

1. Using `torch.nn.RNN`, initialize `W_hh` with a small scale (e.g. multiply by 0.1) and measure the gradient norm reaching the input embedding for sequence lengths 5, 20, 50, 100. Plot gradient norm vs sequence length and observe the trend.
2. Repeat with `W_hh` initialized at a large scale (e.g. multiply by 3.0) instead, and observe the opposite trend.
3. Implement gradient clipping manually (without `torch.nn.utils.clip_grad_norm_`) by computing the total gradient norm across all parameters and rescaling if it exceeds a threshold. Verify it matches PyTorch's built-in version.
4. Design a small synthetic task where the correct output at the last time step depends entirely on the very first input, with several irrelevant time steps in between (a "long-range dependency" task). Train a plain RNN and observe whether it succeeds as you increase the number of irrelevant time steps in between.

## Key Terms

| Term | What it actually means |
|---|---|
| Vanishing gradient | The gradient reaching early time steps shrinking toward zero as sequence length grows, due to repeated multiplication by factors less than 1 |
| Exploding gradient | The gradient reaching early time steps growing very large as sequence length grows, due to repeated multiplication by factors greater than 1 |
| Gradient clipping | Rescaling the gradient to cap its norm at a threshold before applying an optimizer update, preventing catastrophically large updates |
| Long-range dependency | A relationship between an early part of a sequence and a much later part, which vanishing gradients make difficult for a plain RNN to learn |
