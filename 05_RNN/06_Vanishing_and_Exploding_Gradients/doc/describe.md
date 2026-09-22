#  Vanishing and Exploding Gradients


##  The Problem: A Product of Many Terms

Lesson 05 showed that the gradient flowing backward through an RNN passes through a $W_{hh}^\top dz$ multiplication at **every single time step**. For a sequence of length $T$, the gradient reaching the earliest time step has effectively been multiplied by something involving $W_{hh}$ (and the local derivative of $\tanh$) $T$ times in a row. Repeated multiplication is exactly the kind of operation that either shrinks toward zero or grows without bound, depending on the scale of what's being multiplied — and for RNNs, this isn't a rare edge case. It's close to the default behavior for anything beyond short sequences.

---

##  The Concept

###  Isolating the repeated factor

From Lesson 05 , the gradient flowing from $h_t$ back to $h_{t-1}$ is (approximately, treating $W_{hh}$ as the dominant term):

$$\frac{\partial h_t}{\partial h_{t-1}} \approx W_{hh}^\top \, \text{diag}\big(\tanh'(z_t)\big)$$

Chaining this across many steps, the gradient reaching $h_1$ from a loss defined at $h_T$ ( $T-1$ steps back) is approximately a **product** of $T-1$ such matrices:

$$\frac{\partial L}{\partial h_1} \approx \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} \Big[W_{hh}^\top\, \text{diag}\big(\tanh'(z_t)\big)\Big]$$

That product of $T-1$ matrices is the crux of the problem. Two facts about its ingredients determine what happens as $T$ grows:

- $\tanh'(z) = 1 - \tanh(z)^2$ is **at most 1**, and typically much smaller than 1 whenever $z$ is far from 0 — $\tanh$ saturates for large $|z|$ (the CNN module's discussion of saturating nonlinearities applies here too). Multiplying by something $\le 1$ repeatedly, $T-1$ times, tends to shrink the gradient toward zero: **vanishing gradients**.
- If $W_{hh}$'s largest eigenvalue happens to be greater than 1, repeated multiplication by $W_{hh}$ can instead **grow** the gradient — potentially very large — as $T$ increases: **exploding gradients**.

Both are consequences of the *same* repeated-multiplication structure; which one dominates depends on the specific scale of $W_{hh}$ and how saturated the $\tanh$ activations are. In practice, a poorly-behaved RNN can show signs of both at different points in training, or on different parts of the same sequence.

###  Making it concrete: a scalar toy example

To see the exponential trend in actual numbers rather than just an inequality, strip everything down to the 1-dimensional case (a single hidden unit), so the "product of matrices" above becomes an ordinary product of numbers. Suppose $\tanh'(z_t) \approx 0.9$ at every step (a reasonable value when the hidden state isn't deeply saturated), and compare two scalar values of $W_{hh}$:

| $t$ steps back | Cumulative factor, $W_{hh}=0.5$: $(0.5 \times 0.9)^t$ | Cumulative factor, $W_{hh}=1.5$: $(1.5 \times 0.9)^t$ |
|---|---|---|
| 1 | 0.45 | 1.35 |
| 5 | 0.0185 | 4.437 |
| 10 | 0.000342 | 20.11 |
| 20 | $1.17\times 10^{-7}$ | 404.3 |
| 50 | $4.6\times 10^{-18}$ | $3.3\times 10^{6}$ |

With $W_{hh} = 0.5$ (per-step factor $0.45 < 1$), the gradient reaching 50 steps back has shrunk by a factor of roughly $4.6\times 10^{-18}$ — for any practical purpose, exactly zero. With $W_{hh}=1.5$ (per-step factor $1.35 > 1$), the same 50 steps instead **amplify** the gradient by a factor of about 3.3 million. Notice how sharply the two columns diverge even though $W_{hh}$ only differs by a factor of 3 ($0.5$ vs $1.5$) — this is the hallmark of exponential growth/decay: small differences in the per-step multiplier become enormous differences after enough repetitions. This table is exactly what Exercises 1–2 ask you to reproduce empirically with a real RNN, rather than this simplified scalar approximation.

###  Vanishing gradients: the network can't learn long-range dependencies

If the gradient reaching an early time step is effectively zero (as in the $W_{hh}=0.5$ column above, past even 20 steps), weight updates driven by that gradient are also effectively zero. The network cannot learn to use information from early in a long sequence to affect later predictions, no matter how relevant that information actually is.

Concretely: a plain RNN trying to learn that "the subject introduced in word 2 determines the correct verb conjugation in word 40" will struggle, because the gradient connecting word 40's error back to word 2's contribution has vanished by the time it gets there — the mechanism is precisely.2's table, just with 38 steps in between instead of 50. This is the single biggest practical limitation of plain RNNs, and precisely the problem LSTM and GRU (Lessons 07–08) are architected to solve.

###  Exploding gradients: training becomes unstable

An exploding gradient produces a huge weight update in one step, which can throw the network's weights into a wildly different (and usually much worse) region of parameter space, sometimes producing `NaN` values (as seen in the CNN module's training-troubleshooting table) and derailing training entirely.

Unlike vanishing gradients, exploding gradients are comparatively **easy to detect** — the loss suddenly spikes or becomes `NaN` — and have a direct, effective fix, covered next.

###  Gradient clipping: a direct fix for exploding gradients

Before applying the optimizer's update, rescale the gradient if its norm exceeds a threshold, capping how large any single update step can be:

$$\text{if } \|g\| > \theta: \qquad g \leftarrow g \cdot \frac{\theta}{\|g\|}$$

where $g$ is the (flattened) gradient vector and $\theta$ is the chosen threshold. This preserves the gradient's **direction** (still a genuinely useful update signal — it still points toward decreasing loss) while capping its **magnitude**, preventing any single step from being catastrophically large. It's cheap, simple, and standard practice in essentially every RNN training loop:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

**Why this fixes exploding but not vanishing gradients:** clipping only ever *shrinks* an already-too-large gradient down to a manageable size — it has nothing to do when the gradient is already near zero. There's no analogous "un-shrinking" operation, because clipping only has access to the gradient it received; it has no way to know what the "true," un-vanished gradient *should* have been. This is why gradient clipping is standard practice **alongside** architectural fixes (Lessons 07–08), not a substitute for them — it treats one symptom, cheaply, while leaving the other problem (and its underlying cause) completely untouched.

###  The real fix for vanishing gradients: architecture, not training tricks

Gradient clipping, careful weight initialization, and using a less-saturating activation don't solve the fundamental issue that a plain RNN's hidden state is *overwritten* at every step (Lesson 02's point about the hidden state not being a literal buffer, and the direct source of the repeated-multiplication problem ). LSTM (Lesson 07) and GRU (Lesson 08) instead give the network an explicit, learnable mechanism to **preserve** information across many time steps largely unchanged when needed — directly targeting the repeated-multiplication problem at its source, rather than working around its symptoms the way clipping does.

---

##  Use It: Seeing Vanishing and Exploding Gradients Empirically

###  Measuring gradient magnitude vs. sequence length

```python
import torch
import torch.nn as nn

def gradient_norm_at_input(seq_len, hh_scale, hidden_size=8, input_size=1, seed=0):
    torch.manual_seed(seed)
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    with torch.no_grad():
        rnn.weight_hh_l0.mul_(hh_scale)   # rescale W_hh to control the per-step factor 

    x = torch.randn(1, seq_len, input_size, requires_grad=True)
    h0 = torch.zeros(1, 1, hidden_size)
    _, h_final = rnn(x, h0)

    loss = (h_final ** 2).sum()
    loss.backward()

    # Gradient reaching the very first input, x[:, 0, :] -- this is what has
    # traveled backward through every one of the seq_len time steps.
    return x.grad[0, 0, :].norm().item()

for hh_scale in [0.3, 1.0, 3.0]:
    print(f"\nW_hh scaled by {hh_scale}:")
    for T in [5, 20, 50, 100]:
        g = gradient_norm_at_input(T, hh_scale)
        print(f"  T={T:>4}: grad norm at first input = {g:.3e}")
```

Running this reproduces  pattern with a real (8-dimensional hidden state) RNN rather than a hand-simplified scalar: with `hh_scale=0.3`, the printed gradient norm should shrink toward numbers indistinguishable from zero as `T` grows; with `hh_scale=3.0`, it should grow to very large (possibly `inf` or `nan` at `T=100`) values; `hh_scale=1.0` sits in between, and its behavior is a good illustration of why the "does it vanish or explode" outcome is sensitive to the exact scale.

### Gradient clipping in action

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
with torch.no_grad():
    rnn.weight_hh_l0.mul_(3.0)   # deliberately push toward exploding gradients

x = torch.randn(1, 100, 1)
h0 = torch.zeros(1, 1, 8)
_, h_final = rnn(x, h0)
loss = (h_final ** 2).sum()
loss.backward()

total_norm_before = torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=float("inf"))  # measure only
print(f"Gradient norm before clipping: {total_norm_before:.3e}")

torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=5.0)  # actually clip this time
total_norm_after = torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=float("inf"))
print(f"Gradient norm after clipping:  {total_norm_after:.3e}")
```

The first printed norm should be large (possibly enormous, reflecting the `hh_scale=3.0` exploding-gradient setup ); the second should be capped at (at most) `5.0`, confirming the clip actually took effect. Note the trick used here: calling `clip_grad_norm_` with `max_norm=float("inf")` never rescales anything, so it's a convenient way to just *measure* the current gradient norm without modifying it — useful for exactly this kind of before/after comparison.


---

## Key Terms

| Term | What it actually means |
|---|---|
| Vanishing gradient | The gradient reaching early time steps shrinking toward zero as sequence length grows, due to repeated multiplication by factors less than 1 |
| Exploding gradient | The gradient reaching early time steps growing very large as sequence length grows, due to repeated multiplication by factors greater than 1 |
| Gradient clipping | Rescaling the gradient to cap its norm at a threshold before applying an optimizer update, preventing catastrophically large updates — fixes exploding gradients only, not vanishing ones |
| Long-range dependency | A relationship between an early part of a sequence and a much later part, which vanishing gradients make difficult for a plain RNN to learn |