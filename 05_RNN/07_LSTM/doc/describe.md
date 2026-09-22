# 07. LSTM

## Learning Objectives

By the end of this lesson, you should be able to:

- Explain the cell state and why it provides a more direct path for gradients than a plain RNN's hidden state
- Write the equations for all three LSTM gates and the cell state update, and explain what each gate is "deciding"
- Trace a tiny LSTM cell by hand through two time steps, from gate values to the final hidden state
- Implement an LSTM cell from scratch and verify it against `torch.nn.LSTMCell`
- Explain precisely, in terms of the backward-pass equations, why the cell state addresses vanishing gradients where a plain RNN's hidden state cannot

---

## 1. The Problem: Overwriting vs. Preserving

Lesson 06 diagnosed the core issue with plain RNNs: the hidden state is fully **overwritten** at every step by a $\tanh$ of a linear combination (Lesson 03's $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1}+b_h)$), and the repeated multiplication this creates during backpropagation causes gradients — and therefore the ability to learn long-range dependencies — to vanish over long sequences.

Long Short-Term Memory (LSTM), introduced by Hochreiter & Schmidhuber in 1997, fixes this with a structural change: give the network a **second** piece of state — the cell state — specifically designed to let information flow across many time steps with only small, deliberate modifications, rather than being overwritten wholesale every step.

---

## 2. The Concept

### 2.1 Two pieces of state instead of one

A plain RNN has one hidden state, fully recomputed every step. LSTM maintains two:

```
h_t: the hidden state (same role as before -- used for output, passed to next layer)
c_t: the cell state (NEW -- a more protected channel for long-term information)
```

The cell state is the key innovation: rather than being replaced at every step, it's updated through **addition** and elementwise multiplication by values between 0 and 1 (gates). As §2.5 makes precise, this is what allows gradients to flow backward through many steps via addition rather than repeated matrix multiplication — sidestepping the shrink-or-explode dynamic diagnosed in Lesson 06.

### 2.2 Gates: learned, per-step control over what to remember and forget

An LSTM cell has three gates, each a small neural network layer — a linear transformation followed by a **sigmoid** — that controls how information flows into, out of, and through the cell state. Recall $\sigma(z) = \frac{1}{1+e^{-z}}$ squashes any real number to the range $(0, 1)$; this is what makes a gate's output interpretable as "how much of this passes through" (0 = block completely, 1 = pass through entirely):

$$f_t = \sigma\big(W_f [h_{t-1}, x_t] + b_f\big) \qquad \text{forget gate — "how much of the old cell state to keep"}$$

$$i_t = \sigma\big(W_i [h_{t-1}, x_t] + b_i\big) \qquad \text{input gate — "how much of the new candidate to add"}$$

$$o_t = \sigma\big(W_o [h_{t-1}, x_t] + b_o\big) \qquad \text{output gate — "how much of the cell state to expose as } h_t\text{"}$$

$[h_{t-1}, x_t]$ denotes **concatenating** the previous hidden state and the current input into one longer vector before the linear transformation — a common notational shorthand. In practice this is equivalent to (and usually implemented as) two separate weight matrices, one for $h_{t-1}$ and one for $x_t$, added together — mirroring the $W_{hh}$ / $W_{xh}$ split from Lesson 03, just now with three (soon four) separate pairs of such matrices instead of one.

### 2.3 The candidate cell state and the cell state update

$$\tilde{c}_t = \tanh\big(W_c [h_{t-1}, x_t] + b_c\big) \qquad \text{candidate — "new information proposed at this step"}$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \qquad \underbrace{\phantom{xx}}_{\text{keep old (gated)}} \quad \underbrace{\phantom{xx}}_{\text{add new (gated)}}$$

This cell-state update is the central equation of the whole lesson. $\odot$ denotes elementwise multiplication (not matrix multiplication) — the new cell state is a gated blend of the old cell state and a new candidate. If the forget gate $f_t$ is close to 1 and the input gate $i_t$ is close to 0, then $c_t \approx c_{t-1}$: the cell state passes through almost **unchanged**. This is exactly the mechanism that lets relevant information from early in a sequence survive, largely untouched, across many later steps — something a plain RNN's $\tanh$-squashed hidden state has no direct way to do.

### 2.4 The hidden state output

$$h_t = o_t \odot \tanh(c_t)$$

The hidden state is a gated, $\tanh$-squashed *view* of the cell state — what the cell "chooses to expose" at this step. It's used for any output computation (Lesson 04's $y_t = W_{hy}h_t + b_y$) and passed to the next step (and, in a stacked architecture, to the next layer — Lesson 10).

### 2.5 Why this addresses vanishing gradients

This is worth making precise, since it's the entire point of the architecture. Backpropagating through $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, the gradient path back to $c_{t-1}$ is:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad \text{(elementwise)}$$

Compare this directly to Lesson 06 §2.1's plain-RNN backward step, $\partial h_t/\partial h_{t-1} \approx W_{hh}^\top \text{diag}(\tanh'(z_t))$ — a **matrix multiplication by the same fixed $W_{hh}$ at every single step**, which is exactly what produced the exponential shrink-or-explode dynamic in Lesson 06's worked table. Here, the analogous term is just $f_t$ — an elementwise multiplication by a **learned, per-step gate value**, not a fixed matrix reapplied identically everywhere.

The practical consequence: when the network learns to keep $f_t$ close to 1 for information that needs to persist, the gradient chain $\prod_t f_t$ stays close to $1^{(\text{many steps})} \approx 1$ instead of decaying like $(0.45)^{50} \approx 10^{-18}$ (Lesson 06 §2.2's number). The network has an *explicit, learned* mechanism for preserving long-range information, rather than relying on $\tanh$'s derivative happening to stay large by chance.

This doesn't make LSTMs immune to vanishing/exploding gradients entirely — gradient clipping (Lesson 06 §2.5) is still standard practice, and a poorly-trained LSTM can still learn forget-gate values that don't stay near 1 when they should. But it addresses the *structural* cause far more directly than any training trick could, which is why LSTM (and GRU, Lesson 08) rather than clipping-plus-initialization tricks became the standard solution.

---

## 3. A Complete Worked Example: Tracing an LSTM Cell by Hand

To make §2's four equations concrete, let's trace a tiny 1-dimensional LSTM cell ($h_t$ and $c_t$ both scalars, $x_t$ scalar) through two time steps.

### 3.1 Setup

Choose weights that make the arithmetic clean to follow — each gate looks at only one part of $[h_{t-1}, x_t]$:

$$W_f = [1, 0],\ b_f=0 \quad\Rightarrow\quad z_f = h_{t-1}$$
$$W_i = [0, 1],\ b_i=0 \quad\Rightarrow\quad z_i = x_t$$
$$W_o = [1, 1],\ b_o=0 \quad\Rightarrow\quad z_o = h_{t-1} + x_t$$
$$W_c = [1, -1],\ b_c=0 \quad\Rightarrow\quad z_c = h_{t-1} - x_t$$

Start from $h_0 = 0,\ c_0 = 0$, with inputs $x_1 = 1.0,\ x_2=-1.0$ — the same input sequence used throughout Lessons 03–06, so you can compare how differently an LSTM behaves on it.

### 3.2 Step $t=1$

$$z_f = h_0 = 0 \;\Rightarrow\; f_1 = \sigma(0) = 0.500$$
$$z_i = x_1 = 1.0 \;\Rightarrow\; i_1 = \sigma(1.0) \approx 0.731$$
$$z_o = h_0+x_1 = 1.0 \;\Rightarrow\; o_1 = \sigma(1.0) \approx 0.731$$
$$z_c = h_0-x_1 = -1.0 \;\Rightarrow\; \tilde{c}_1 = \tanh(-1.0) \approx -0.762$$

$$c_1 = f_1 c_0 + i_1 \tilde{c}_1 = (0.5)(0) + (0.731)(-0.762) \approx -0.557$$
$$h_1 = o_1 \tanh(c_1) = (0.731)\tanh(-0.557) \approx (0.731)(-0.506) \approx -0.370$$

### 3.3 Step $t=2$

$$z_f = h_1 \approx -0.370 \;\Rightarrow\; f_2 = \sigma(-0.370) \approx 0.409$$
$$z_i = x_2 = -1.0 \;\Rightarrow\; i_2 = \sigma(-1.0) \approx 0.269$$
$$z_o = h_1+x_2 \approx -1.370 \;\Rightarrow\; o_2 = \sigma(-1.370) \approx 0.203$$
$$z_c = h_1-x_2 \approx 0.630 \;\Rightarrow\; \tilde{c}_2 = \tanh(0.630) \approx 0.559$$

$$c_2 = f_2 c_1 + i_2 \tilde{c}_2 \approx (0.409)(-0.557) + (0.269)(0.559) \approx -0.228 + 0.150 \approx -0.077$$
$$h_2 = o_2 \tanh(c_2) \approx (0.203)\tanh(-0.077) \approx (0.203)(-0.077) \approx -0.016$$

**What to notice:** the forget gate values ($f_1=0.5$, $f_2\approx 0.409$) are both moderate here, since these particular toy weights weren't chosen to demonstrate "mostly remember." Exercise 4 asks you to redo this same trace with weights chosen specifically so $f_t$ stays close to 1, so you can watch $c_t \approx c_{t-1}$ hold almost exactly, step after step — the behavior §2.3 describes in words.

---

## 4. Use It: Code

### 4.1 From-scratch implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

### 4.2 Reproducing §3's numbers

```python
params = {
    "W_f": np.array([1.0, 0.0]), "b_f": np.array([0.0]),
    "W_i": np.array([0.0, 1.0]), "b_i": np.array([0.0]),
    "W_o": np.array([1.0, 1.0]), "b_o": np.array([0.0]),
    "W_c": np.array([1.0, -1.0]), "b_c": np.array([0.0]),
}

h, c = np.array([0.0]), np.array([0.0])
for t, x_val in enumerate([1.0, -1.0], start=1):
    h, c = lstm_cell_forward(np.array([x_val]), h, c, params)
    print(f"t={t}: h_t = {h}, c_t = {c}")
```

```
t=1: h_t = [-0.3699], c_t = [-0.5569]
t=2: h_t = [-0.0156], c_t = [-0.0772]
```

These match §3's hand-worked values, confirming the implementation.

### 4.3 Cross-checking against `torch.nn.LSTMCell`

```python
import torch
import torch.nn as nn

cell = nn.LSTMCell(input_size=1, hidden_size=1)

# torch.nn.LSTMCell concatenates all four gates' weights into single matrices,
# in the fixed order [input, forget, cell (candidate), output] -- check the
# documentation before copying weights, since this order is easy to get wrong.
with torch.no_grad():
    # weight_ih / weight_hh have shape (4*hidden_size, input_size / hidden_size)
    cell.weight_ih.copy_(torch.tensor([[0.0], [1.0], [1.0], [1.0]]))   # i, f, c, o vs. x_t
    cell.weight_hh.copy_(torch.tensor([[1.0], [1.0], [-1.0], [1.0]]))  # i, f, c, o vs. h_(t-1)
    cell.bias_ih.zero_()
    cell.bias_hh.zero_()

h, c = torch.zeros(1, 1), torch.zeros(1, 1)
for t, x_val in enumerate([1.0, -1.0], start=1):
    x_t = torch.tensor([[x_val]])
    h, c = cell(x_t, (h, c))
    print(f"t={t}: h_t = {h.item():.4f}, c_t = {c.item():.4f}")
```

Running this should reproduce §4.2's numbers (up to rounding and the gate-ordering details called out in the comment) — a good exercise in reading unfamiliar library documentation carefully, which Exercise 1 asks you to do in full.

### 4.4 A practical initialization trick, previewed

Exercise 2 below asks you to investigate initializing $b_f$ to a large positive value (e.g., 5) before training. As a preview of why: $\sigma(5) \approx 0.993$, meaning the forget gate starts training already close to "fully remember," rather than at the neutral $\sigma(0)=0.5$ this section's toy example used. This is a well-known, simple trick for helping LSTMs learn long-range dependencies faster, especially early in training before the gates have learned anything useful on their own.

---

## Exercises

**Reproducing and verifying**

1. Implement `lstm_cell_forward` (§4.1) and verify it matches `torch.nn.LSTMCell` exactly, given the same weights (§4.3 gets you started — read PyTorch's documentation for `LSTMCell` carefully to confirm the exact gate ordering `[i, f, c, o]` used when concatenating weights, since getting this order wrong is a very easy mistake that still runs without error).

**Understanding the gates**

2. Set the forget gate's bias $b_f$ to a large positive value (e.g., 5) before training, and explain — using the sigmoid function and §4.4's preview — why this is a common practical initialization trick. What does `sigmoid(5)` evaluate to, and what does that value mean for how much of $c_{t-1}$ survives at the very start of training, before any weights have been learned?
3. Reproduce the gradient-magnitude-vs-sequence-length comparison from Lesson 06 (§3.1 there), this time comparing a plain `nn.RNN` against an `nn.LSTM` of the same hidden size, on the same sequence lengths. Confirm the LSTM's gradient stays substantially larger at long sequence lengths, and connect what you observe back to §2.5's $\partial c_t/\partial c_{t-1} = f_t$ argument.

**Tracing by hand**

4. Redo §3's hand trace with new weights chosen so that $f_t$ stays close to 1 at every step (for example, set $W_f = [0,0]$ and $b_f = 5$, so $z_f = 5$ regardless of $h_{t-1}$ and $x_t$, giving $f_t = \sigma(5) \approx 0.993$ at every step). Keep the other three gates' weights the same as §3.1. Trace $c_t$ across 3 steps and confirm it changes much more slowly than in §3 — i.e., that $c_t \approx c_{t-1} + i_t\tilde{c}_t$ (since $f_t \approx 1$) rather than the more balanced blend seen in §3.

---

## Key Terms

| Term | What it actually means |
|---|---|
| LSTM (Long Short-Term Memory) | An RNN variant with an explicit cell state and gating mechanism, designed to preserve long-range information and address vanishing gradients |
| Cell state ($c_t$) | LSTM's protected channel for long-term information, updated through gated addition rather than full overwriting at every step |
| Forget gate ($f_t$) | The gate controlling how much of the previous cell state is retained at the current step |
| Input gate ($i_t$) | The gate controlling how much new candidate information is added to the cell state at the current step |
| Output gate ($o_t$) | The gate controlling how much of the cell state is exposed as the hidden state at the current step |
| Candidate cell state ($\tilde{c}_t$) | The new information proposed at the current step, before being gated into the cell state by the input gate |
| Sigmoid ($\sigma$) | The function $\sigma(z) = 1/(1+e^{-z})$, squashing any real number to $(0,1)$; used in every LSTM gate so its output can be read as "how much passes through" |