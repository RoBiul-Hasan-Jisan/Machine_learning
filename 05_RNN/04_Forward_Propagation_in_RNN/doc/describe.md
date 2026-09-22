#  Forward Propagation in RNN



##  The Problem: From One Step to a Whole Sequence

Lesson 03 nailed down exactly what happens at *one* time step:

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

But a sequence has $T$ steps, not one. **Forward propagation** is running that single-step computation repeatedly — once per element of the sequence — feeding each step's hidden state into the next, and reading out outputs at whichever steps the task actually requires. This is the mechanical process behind Lesson 01's "same RNN cell reused at every time step" diagram, and it's the last piece needed before Lesson 05 can talk about how gradients flow *backward* through the same structure.

---

## The Concept: Unrolling the Cell Across Time

### The unrolled forward pass

```
h_0 = zeros

for t in 1..T:
    h_t = tanh(W_xh @ x_t + W_hh @ h_(t-1) + b_h)     <- Lesson 03's cell equation, unchanged
    y_t = W_hy @ h_t + b_y                              <- optional, only if this step needs an output

return all h_t (and/or all y_t), and the final h_T
```

Calling this "unrolling" is a useful mental picture: applying the *same* cell $T$ times produces a computation graph that looks like $T$ copies of the same small network laid out side by side, each one feeding its output hidden state into the next copy's input. That's exactly Lesson 01's diagram, now backed by the concrete equation from Lesson 03. This unrolled view matters directly for Lesson 05 (backpropagation through time): gradients have to flow backward through every one of these $T$ copies, one at a time, which is where several of the practical difficulties with training RNNs come from.

### What differs between the three patterns

Every pattern below uses the **exact same** hidden-state update from  — nothing changes about how $h_t$ is computed. What differs is only:

1. **Where $x_t$ comes from** at each step (a real element of the input, nothing, or the previous step's own output), and
2. **Where $y_t$ is read out** (every step, only the last step, or only after the input is fully consumed).

Keeping this in mind will help you see the three patterns as variations on one idea, not three different algorithms.

---

##  Many-to-One: Read the Output Only at the Last Step

### The idea

For tasks like sentiment classification — a whole review maps to one label — only the *final* hidden state matters for the prediction. Every earlier hidden state exists purely to build up to it; nothing is read out along the way.

```
x_1 -> h_1 -> x_2 -> h_2 -> ... -> x_T -> h_T -> y  (single output, read from h_T only)
```

###  Code

```python
import numpy as np

def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

def rnn_forward_many_to_one(sequence, W_xh, W_hh, b_h, W_hy, b_y):
    h = np.zeros(W_hh.shape[0])
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
    y = W_hy @ h + b_y     # only computed once, after the loop
    return y
```
### Tracing a Tiny Example by Hand

Reusing **Lesson 03** parameters:
| **Parameter** | **Value** |
|---|---:|
| `dx` | `1` |
| `dh` | `2` |
| `W_{xh}` | 2 × 1 |
| `W_{hh}` | 2 × 2 |
| `b_h` | 2 × 1 |

### Weight Matrix

$$
W_{xh} =
\begin{bmatrix}
1.0 \\\\
0.5
\end{bmatrix}
$$

$$
W_{hh} =
\begin{bmatrix}
0.5 & 0.0 \\\\
0.0 & 0.5
\end{bmatrix}
$$

$$
b_h =
\begin{bmatrix}
0 \\\\
0
\end{bmatrix}
$$

With the sequence:

$$
x_1 = 1.0,\qquad x_2 = -1.0
$$

we already computed :

$$
h_1 \approx
\begin{bmatrix}
0.762 \\
0.462
\end{bmatrix},
\qquad
h_2 \approx
\begin{bmatrix}
-0.550 \\
-0.263
\end{bmatrix}
$$

For a **many-to-one** RNN, the recurrence ends at the final hidden state. Since $T=2$:

$$
h_T = h_2
$$

Now add a **readout layer**:

$$
W_{hy} =
\begin{bmatrix}
1.0 & -1.0
\end{bmatrix}
$$

where $W_{hy}$ has shape $(1,2)$ and produces a single scalar output. Let:

$$
b_y = 0
$$

The output is:

$$
\begin{aligned}
y
&= W_{hy}h_2 + b_y \\
&= (1.0)(-0.550) + (-1.0)(-0.263) \\
&= -0.550 + 0.263 \\
&= -0.287
\end{aligned}
$$

Therefore:

$$
\boxed{y \approx -0.287}
$$

> **Key idea:** $y$ never looked at $h_1$ directly. It only used the final hidden state $h_2$. However, $h_2$ was influenced by $h_1$ through the recurrence.

So the information flow is:

$$
x_1 \rightarrow h_1 \rightarrow h_2 \rightarrow y
$$

and

$$
x_2 \rightarrow h_2 \rightarrow y
$$

This is the core idea of **many-to-one RNNs**: information from all previous time steps is progressively folded into the final hidden state, and the final hidden state is then used to produce the output.

## Many-to-Many: Read an Output at Every Step

###  The idea

For tasks like part-of-speech tagging (every word gets its own label) or language modeling (predict the next token at every position), an output is read out at **every** time step, not just the last one:

```
x_1 -> h_1 -> y_1
x_2 -> h_2 -> y_2
...
x_T -> h_T -> y_T
```

### Code

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

The only structural change from  many-to-one version: `y_t` is computed and stored **inside** the loop, at every step, rather than once after it.

###  A second many-to-many variant: encoder-decoder

There's a second flavor of many-to-many where the entire input is consumed *first*, and outputs are only produced *afterward*, at a potentially **different number of steps** than the input had — e.g., translating a 5-word English sentence into a 7-word French sentence. This can't be handled by simply reading out $y_t$ at every input step, since the number of outputs doesn't match the number of inputs. It needs an additional idea — two separate RNNs, one that reads the input and one that generates the output — covered specifically in Lesson 11 (Sequence-to-Sequence Models). The many-to-many pattern  is the simpler case where input length and output length always match.

---

##  One-to-Many: A Single Input Drives an Entire Generated Sequence

###  The idea

For tasks like image captioning — one image produces a whole sentence — a single input is used to initialize the hidden state (or is fed in only at the first step), and the network then **generates** a sequence of outputs, typically feeding each step's own output back in as the next step's input:

```
x -> h_1 -> y_1 -> h_2 -> y_2 -> h_3 -> y_3 -> ...
        (y_1 fed back in as input to produce h_2, and so on)
```
---
###  Code

```python
def rnn_forward_one_to_many(x_initial, num_steps, W_xh, W_hh, b_h, W_hy, b_y):
    h = np.zeros(W_hh.shape[0])
    outputs = []
    x_t = x_initial              # real input, only used at the very first step
    for _ in range(num_steps):
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        y_t = W_hy @ h + b_y
        outputs.append(y_t)
        x_t = y_t                # feed this step's own output back in as the next input
    return outputs
```

Compare this carefully : the difference isn't in how $h_t$ is computed (identical), but in where $x_t$ comes from at each step — a real, externally-given input in many-to-many, versus the network's *own previous output* here. This exact feed-output-back-in-as-input mechanism reappears centrally in Lesson 14 (Text Generation), where it's precisely how a trained language model produces text one token at a time: predict a word, feed that word back in, predict the next word, and so on.

---

##  Batched Forward Propagation

In practice, forward propagation runs on a **batch** of sequences at once — Lesson 02's padded/masked batches — not one sequence at a time. Every quantity simply picks up an extra batch dimension:

| Quantity | Single sequence | Batched |
|---|---|---|
| $x_t$ | $(d_x,)$ | $(\text{batch}, d_x)$ |
| $h_t$ | $(d_h,)$ | $(\text{batch}, d_h)$ |
| $y_t$ | $(d_y,)$ | $(\text{batch}, d_y)$ |

The same weight matrices ($W_{xh}$, $W_{hh}$, $b_h$, and so on) are applied identically across every sequence in the batch, in parallel, at each time step — nothing about the *weights* changes; only the input tensors grow a leading batch axis. This is exactly what `torch.nn.RNN` does internally when given a `(batch, T, input_size)` tensor: it's running the loop from  once, but with every step's arithmetic operating on a whole batch simultaneously via matrix multiplication, instead of looping over the batch dimension too.

---

## Use It: Verifying Against `torch.nn.RNN`

```python
import numpy as np
import torch
import torch.nn as nn

# Reuse Lesson 03's parameters
W_xh = np.array([[1.0], [0.5]])
W_hh = np.array([[0.5, 0.0], [0.0, 0.5]])
b_h  = np.array([0.0, 0.0])
W_hy = np.array([[1.0, -1.0]])
b_y  = np.array([0.0])

sequence = [np.array([1.0]), np.array([-1.0])]

# --- From-scratch many-to-many ---
def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

def rnn_forward_many_to_many(sequence, W_xh, W_hh, b_h, W_hy, b_y):
    h = np.zeros(W_hh.shape[0])
    outputs, hiddens = [], []
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        hiddens.append(h.copy())
        outputs.append(W_hy @ h + b_y)
    return outputs, hiddens

outputs, hiddens = rnn_forward_many_to_many(sequence, W_xh, W_hh, b_h, W_hy, b_y)
print("From-scratch hidden states:", hiddens)

# --- torch.nn.RNN, loaded with the same weights ---
rnn = nn.RNN(input_size=1, hidden_size=2, batch_first=True)
with torch.no_grad():
    rnn.weight_ih_l0.copy_(torch.tensor(W_xh, dtype=torch.float32))
    rnn.weight_hh_l0.copy_(torch.tensor(W_hh, dtype=torch.float32))
    rnn.bias_ih_l0.zero_()
    rnn.bias_hh_l0.zero_()

x_batch = torch.tensor([[[1.0], [-1.0]]])   # shape (batch=1, T=2, input_size=1)
h0 = torch.zeros(1, 1, 2)                    # shape (num_layers=1, batch=1, hidden_size=2)
all_hidden, final_hidden = rnn(x_batch, h0)
print("torch.nn.RNN hidden states:", all_hidden.detach().numpy())
```

Running this, `hiddens` from the from-scratch loop and `all_hidden` from `torch.nn.RNN` should match (up to floating-point rounding) at every time step, and both should match Lesson 03's hand-worked $h_1 \approx [0.762, 0.462]$ and $h_2 \approx [-0.550, -0.263]$. This three-way agreement — hand calculation, from-scratch code, and PyTorch's built-in layer — is exactly the kind of check worth running whenever you're not sure your implementation of an equation is correct.

---

## Exercises

**Implementing the three patterns**

1. Implement `rnn_forward_many_to_one` and verify its final hidden state matches `torch.nn.RNN`'s final hidden state output, given the same weights and the same input sequence as.
2. Implement `rnn_forward_many_to_many`  and verify every step's hidden state matches the corresponding position in `torch.nn.RNN`'s full output sequence (not just the final one) — this is exactly what  does; extend it to also check the `y_t` outputs, not just the hidden states.
3. Implement `rnn_forward_one_to_many`  with a toy 1-dimensional input/output, and run it for 5 steps starting from some `x_initial`. Print every `y_t` and confirm each one differs from the last (if they don't, check whether your weights collapsed into one of Lesson 03  degenerate cases).


---

## Key Terms

| Term | What it actually means |
|---|---|
| Forward propagation (in an RNN) | Repeatedly applying the RNN cell equation across every time step of a sequence, feeding each hidden state into the next |
| Unrolling | Viewing an RNN's repeated application of the same cell across time steps as a chain of $T$ copies, useful for reasoning about both forward and backward computation |
| Many-to-one | An RNN usage pattern where a whole input sequence produces a single output, read from the final hidden state only |
| Many-to-many | An RNN usage pattern where an output is produced at every time step (or, in the encoder-decoder variant, after the full input is consumed, at a possibly different number of steps) |
| One-to-many | An RNN usage pattern where a single input drives the generation of an entire output sequence, often by feeding each output back in as the next input |
| Batched forward propagation | Running the same forward-propagation loop with every quantity carrying an extra leading batch dimension, so many sequences are processed in parallel |