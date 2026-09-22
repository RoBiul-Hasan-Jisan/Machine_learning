#  Backpropagation Through Time



##  The Problem: Training Means Differentiating Through Every Reuse of the Same Weights

Lesson 04's forward pass reuses the exact same weights ($W_{xh}$, $W_{hh}$, $b_h$) at every time step. Training the network means computing the gradient of the loss with respect to those weights — but each weight influenced the loss through **every** time step it participated in, not just one. Backpropagation Through Time (BPTT) is the specific application of the chain rule that correctly accounts for this repeated use of the same weights across the unrolled sequence.

---

##  The Concept

###  Unrolling makes BPTT look like ordinary backprop, with one twist

Lesson 04's "unrolling" picture — treating the RNN as $T$ chained copies of the same cell — made forward propagation easy to reason about. Backpropagation through that same unrolled graph works exactly like backpropagation through any deep feedforward network (covered generally in the CNN module's Lesson 09): gradients flow backward from the loss, through each time step, via the chain rule.

**The twist:** since the *same* $W_{hh}$ (and $W_{xh}$, $b_h$) was used at every one of the $T$ steps, the total gradient for $W_{hh}$ is the **sum** of its contribution at every individual time step, not just the contribution from the last one:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \left(\frac{\partial L}{\partial h_t}\right)\left(\frac{\partial h_t}{\partial W_{hh}}\bigg|_{\text{direct, at step } t}\right)$$

Here "direct, at step $t$" means: treat $h_{t-1}$ as a fixed number at this point in the sum, and only differentiate $h_t$'s own formula $\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ with respect to $W_{hh}$ as it appears explicitly in *that* equation. Each term in the sum captures how much $W_{hh}$ mattered at that one specific step; adding them all up gives the total effect of changing $W_{hh}$ everywhere it was used.

This is the same principle as a convolutional filter's gradient in the CNN module (Lesson 09 there): a shared weight's total gradient sums its contribution across every place it was used — spatial positions for a conv filter, time steps here.

###  The gradient flowing backward through the hidden state

At each step going backward, the gradient with respect to $h_t$ has **two sources**:

1. The direct loss contribution at that step, if $y_t$ was read out and feeds into the loss (per Lesson 04's many-to-many pattern), and
2. The gradient flowing back from $h_{t+1}$, since $h_t$ was itself used to compute $h_{t+1}$.

$$\frac{\partial L}{\partial h_t} = \underbrace{\frac{\partial L_t}{\partial h_t}}_{\text{direct, if } y_t \text{ exists}} \;+\; \underbrace{\frac{\partial L}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}}_{\text{via the next step}}$$

Working this out at every step, from $t=T$ backward to $t=1$, and accumulating each step's contribution to $\partial L/\partial W_{hh}$, $\partial L/\partial W_{xh}$, and $\partial L/\partial b_h$ along the way, is exactly what BPTT does. In real projects you never derive or implement this by hand — `loss.backward()` in PyTorch handles it automatically, walking the unrolled computation graph exactly as described — but working through the mechanism concretely  is what makes the gradient problems in Lesson 06 (which arise directly from this backward chain through many time steps) make sense rather than feeling like an unexplained fact about RNNs.

---

##  A Complete Worked Example: BPTT by Hand

Let's trace an entire backward pass, from a loss all the way down to every weight's gradient, using the same tiny setup from Lesson 03 and Lesson 04  — so the forward-pass numbers are already familiar.

### Setup

$$W_{xh} = \begin{bmatrix}1.0\\0.5\end{bmatrix}, \quad W_{hh} = \begin{bmatrix}0.5 & 0.0\\0.0 & 0.5\end{bmatrix}, \quad b_h = \begin{bmatrix}0\\0\end{bmatrix}, \quad h_0 = \begin{bmatrix}0\\0\end{bmatrix}$$

Inputs $x_1 = 1.0,\ x_2 = -1.0$. From Lesson 03  the forward pass gave us:

$$h_1 \approx \begin{bmatrix}0.762\\0.462\end{bmatrix}, \qquad h_2 \approx \begin{bmatrix}-0.550\\-0.263\end{bmatrix}$$

For this example, use a simple loss defined directly on the final hidden state (skipping the output layer, to keep the arithmetic short and focused purely on the recurrence): $L = \tfrac{1}{2}\|h_2\|^2$, whose gradient is simply $\partial L/\partial h_2 = h_2$. (A real task would instead define $L$ in terms of $y_2 = W_{hy}h_2 + b_y$ and a label; the *mechanics* of backpropagating through the recurrence itself are identical either way — only the very first gradient, $\partial L/\partial h_T$, would look different.)

###  Step $t=2$ (the last step, working backward)

Since $L$ depends on $h_2$ directly and there is no $t=3$, the gradient into $h_2$ is just:

$$\frac{\partial L}{\partial h_2} = h_2 \approx \begin{bmatrix}-0.550\\-0.263\end{bmatrix}$$

Recall $\tanh'(z) = 1 - \tanh(z)^2 = 1 - h_t^2$ (elementwise). Using $h_2^2 \approx [0.3025,\ 0.0692]$:

$$dz_2 = \frac{\partial L}{\partial h_2} \odot (1 - h_2^2) \approx \begin{bmatrix}-0.550\\-0.263\end{bmatrix} \odot \begin{bmatrix}0.6975\\0.9308\end{bmatrix} \approx \begin{bmatrix}-0.384\\-0.245\end{bmatrix}$$

($\odot$ denotes elementwise multiplication.) From $dz_2$, the contributions at this step to each parameter's gradient are:

$$dW_{xh}^{(t=2)} = dz_2 \cdot x_2^\top \approx \begin{bmatrix}-0.384\\-0.245\end{bmatrix}(-1.0) = \begin{bmatrix}0.384\\0.245\end{bmatrix}$$

$$dW_{hh}^{(t=2)} = dz_2\, h_1^\top \approx \begin{bmatrix}-0.384\\-0.245\end{bmatrix}\begin{bmatrix}0.762 & 0.462\end{bmatrix} \approx \begin{bmatrix}-0.293 & -0.177\\-0.187 & -0.113\end{bmatrix}$$

$$db_h^{(t=2)} = dz_2 \approx \begin{bmatrix}-0.384\\-0.245\end{bmatrix}$$

And the gradient to pass further backward, into $h_1$:

$$\frac{\partial L}{\partial h_1}\bigg|_{\text{via } t=2} = W_{hh}^\top dz_2 \approx \begin{bmatrix}0.5 & 0\\0 & 0.5\end{bmatrix}\begin{bmatrix}-0.384\\-0.245\end{bmatrix} = \begin{bmatrix}-0.192\\-0.122\end{bmatrix}$$

###  Step $t=1$

Here $h_1$ has **no direct loss contribution** (we only put the loss on $h_2$), so per  formula, the total gradient into $h_1$ is *only* the "via the next step" term computed above:

$$\frac{\partial L}{\partial h_1} \approx \begin{bmatrix}-0.192\\-0.122\end{bmatrix}$$

Using $h_1^2 \approx [0.581,\ 0.213]$:

$$dz_1 = \frac{\partial L}{\partial h_1} \odot (1-h_1^2) \approx \begin{bmatrix}-0.192\\-0.122\end{bmatrix} \odot \begin{bmatrix}0.419\\0.787\end{bmatrix} \approx \begin{bmatrix}-0.080\\-0.096\end{bmatrix}$$

$$dW_{xh}^{(t=1)} = dz_1 \cdot x_1^\top \approx \begin{bmatrix}-0.080\\-0.096\end{bmatrix}(1.0) = \begin{bmatrix}-0.080\\-0.096\end{bmatrix}$$

$$dW_{hh}^{(t=1)} = dz_1\, h_0^\top = \begin{bmatrix}-0.080\\-0.096\end{bmatrix}\begin{bmatrix}0 & 0\end{bmatrix} = \begin{bmatrix}0 & 0\\0 & 0\end{bmatrix}$$

$$db_h^{(t=1)} = dz_1 \approx \begin{bmatrix}-0.080\\-0.096\end{bmatrix}$$

**Worth pausing on:** $dW_{hh}^{(t=1)}$ came out to exactly zero. This isn't a mistake — it's a direct, visible consequence of $h_0 = 0$: since $dW_{hh}^{(t)} = dz_t\, h_{t-1}^\top$, and $h_0 = 0$ here, the very first step can never contribute to $W_{hh}$'s gradient, no matter what $dz_1$ is. If $h_0$ had been nonzero (or learned, as Lesson 03  mentioned), this term wouldn't vanish. This is exactly the kind of thing Exercise 2 asks you to explore further.

###  Summing across time steps

The *total* gradient for each shared parameter is the sum of its per-step contributions:

$$dW_{xh} = dW_{xh}^{(t=1)} + dW_{xh}^{(t=2)} \approx \begin{bmatrix}-0.080\\-0.096\end{bmatrix} + \begin{bmatrix}0.384\\0.245\end{bmatrix} = \begin{bmatrix}0.304\\0.149\end{bmatrix}$$

$$dW_{hh} = dW_{hh}^{(t=1)} + dW_{hh}^{(t=2)} \approx \begin{bmatrix}0 & 0\\0 & 0\end{bmatrix} + \begin{bmatrix}-0.293 & -0.177\\-0.187 & -0.113\end{bmatrix} = \begin{bmatrix}-0.293 & -0.177\\-0.187 & -0.113\end{bmatrix}$$

$$db_h = db_h^{(t=1)} + db_h^{(t=2)} \approx \begin{bmatrix}-0.080\\-0.096\end{bmatrix} + \begin{bmatrix}-0.384\\-0.245\end{bmatrix} = \begin{bmatrix}-0.464\\-0.341\end{bmatrix}$$

These three sums are the actual gradients an optimizer would use to update $W_{xh}$, $W_{hh}$, and $b_h$. Notice that $dW_{hh}$'s two per-step contributions were quite different from each other (one was exactly zero, the other wasn't) — a concrete illustration of  claim that "the same weight, different time steps, generally contributes different amounts to the total gradient."

---

##  Why This Is Expensive for Long Sequences

BPTT's cost — both compute and memory — scales with sequence length $T$, since every one of the $T$ unrolled steps needs its intermediate values (each $h_t$, and whatever else feeds into the local derivatives above) cached for the backward pass to use — the same "cache" concept from the CNN module's Lesson 08, here spanning time steps instead of layers.  computing $dW_{hh}^{(t=1)}$ needed $h_0$, and $dW_{hh}^{(t=2)}$ needed $h_1$ — every hidden state from the forward pass has to be kept around until the backward pass reaches that step.

This is the direct reason very long sequences are often truncated (Lesson 02) or processed with **truncated BPTT**: splitting a long sequence into shorter chunks, running full forward/backward passes within each chunk, and carrying the hidden state (but *not* the full gradient history) across chunk boundaries. Truncated BPTT trades some gradient accuracy — the network can no longer learn dependencies that span across a chunk boundary, since gradients stop flowing backward at that boundary — for tractable memory and compute on long sequences.

---

##  Implementing BPTT from Scratch

```python
import numpy as np

def rnn_backward_step(dh_t, h_t, h_prev, x_t, W_hh):
    """
    One step of BPTT: given the gradient flowing INTO h_t from the future
    (dh_t = dL/dh_t, already combining both sources from Eq. ),
    compute this step's contribution to each parameter's gradient, and the
    gradient to pass further backward (into h_(t-1)).
    """
    dz = dh_t * (1 - h_t ** 2)          # tanh'(z) = 1 - tanh(z)^2 = 1 - h_t^2
    dW_xh = np.outer(dz, x_t)
    dW_hh = np.outer(dz, h_prev)
    db_h = dz
    dh_prev = W_hh.T @ dz               # gradient to pass to the PREVIOUS time step
    return dW_xh, dW_hh, db_h, dh_prev
```

Called once per time step, working backward from $t=T$ to $t=1$, accumulating `dW_xh`, `dW_hh`, `db_h` across every call (since the same weights are shared across steps) and threading `dh_prev` from one call into the next call's `dh_t` — this loop, run over the full unrolled sequence, **is** BPTT in its entirety. There's nothing more to it beyond repeating this one function call $T$ times in reverse order.

---

##  Use It: Reproducing the Hand Calculation, Then Verifying Against Autograd

###  Reproducing  numbers

```python
import numpy as np

W_xh = np.array([[1.0], [0.5]])
W_hh = np.array([[0.5, 0.0], [0.0, 0.5]])
b_h  = np.array([0.0, 0.0])

def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

def rnn_backward_step(dh_t, h_t, h_prev, x_t, W_hh):
    dz = dh_t * (1 - h_t ** 2)
    dW_xh = np.outer(dz, x_t)
    dW_hh = np.outer(dz, h_prev)
    db_h = dz
    dh_prev = W_hh.T @ dz
    return dW_xh, dW_hh, db_h, dh_prev

# --- Forward pass, caching every hidden state ---
inputs = [np.array([1.0]), np.array([-1.0])]
h = np.array([0.0, 0.0])
hiddens = [h]
for x_t in inputs:
    h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
    hiddens.append(h)
# hiddens = [h_0, h_1, h_2]

# --- Backward pass: L = 0.5 * ||h_2||^2, so dL/dh_2 = h_2 ---
dh_next = hiddens[-1].copy()          # dL/dh_2
dW_xh_total = np.zeros_like(W_xh)
dW_hh_total = np.zeros_like(W_hh)
db_h_total = np.zeros_like(b_h)

for t in reversed(range(len(inputs))):   # t = 1, then t = 0  (0-indexed: step "t+1" in the write-up)
    x_t = inputs[t]
    h_t = hiddens[t + 1]
    h_prev = hiddens[t]
    dW_xh, dW_hh, db_h_step, dh_prev = rnn_backward_step(dh_next, h_t, h_prev, x_t, W_hh)
    dW_xh_total += dW_xh
    dW_hh_total += dW_hh
    db_h_total += db_h_step
    dh_next = dh_prev   # only relevant for steps before this one; unused after t=0

print("dW_xh_total:\n", dW_xh_total)
print("dW_hh_total:\n", dW_hh_total)
print("db_h_total:\n", db_h_total)
```

```
dW_xh_total:
 [[0.304]
 [0.149]]
dW_hh_total:
 [[-0.293 -0.177]
 [-0.187 -0.113]]
db_h_total:
 [-0.464 -0.341]
```

These match  hand-worked totals — confirming the implementation is faithful to the derivation.

###  Cross-checking against PyTorch autograd

```python
import torch

W_xh_t = torch.tensor(W_xh, dtype=torch.float32, requires_grad=True)
W_hh_t = torch.tensor(W_hh, dtype=torch.float32, requires_grad=True)
b_h_t  = torch.tensor(b_h, dtype=torch.float32, requires_grad=True)

h = torch.zeros(2)
for x_val in [1.0, -1.0]:
    x_t = torch.tensor([x_val], dtype=torch.float32)
    h = torch.tanh(W_xh_t @ x_t + W_hh_t @ h + b_h_t)

loss = 0.5 * (h ** 2).sum()
loss.backward()

print("dW_xh (autograd):\n", W_xh_t.grad)
print("dW_hh (autograd):\n", W_hh_t.grad)
print("db_h (autograd):\n", b_h_t.grad)
```

Running this should reproduce the same three gradients as  (up to floating-point rounding) — a three-way confirmation, exactly like Lesson 04  : hand calculation, from-scratch BPTT, and PyTorch autograd all agree.

---

## Exercises

**Reproducing and extending the worked example**

1. Implement `rnn_backward_step`  and manually run it backward over the 2-step sequence , reproducing  totals. Then extend the sequence to 4 steps (pick any two additional inputs $x_3, x_4$) and verify the accumulated `dW_hh` against `torch.autograd`'s `.grad` on `W_hh` for the equivalent forward computation.
2. For the same 4-step sequence, isolate and print just the *contribution* to `dW_hh` from each individual time step, before summing them. Confirm they differ from each other in general — and confirm whether the first step's contribution is still exactly zero   (it will be, as long as $h_0 = 0$; try changing $h_0$ to a nonzero vector and see whether that's still true).

**Truncated BPTT**

3. Implement a simple version of truncated BPTT: split an 8-step sequence into two 4-step chunks, carry only the hidden state (not gradient history) across the chunk boundary, and compare the resulting `dW_hh` to the gradient from full (untruncated) BPTT over all 8 steps. Are they the same? Should they be, given  description of what truncated BPTT gives up?
4. Time how BPTT's forward+backward cost scales as you increase sequence length from 10 to 100 to 1000 steps, using `torch.nn.RNN` and `loss.backward()`, and relate the scaling you observe to the caching requirement described .

---

## Key Terms

| Term | What it actually means |
|---|---|
| Backpropagation Through Time (BPTT) | Backpropagation applied to an RNN's unrolled computation graph, where a shared weight's total gradient sums its contribution across every time step it was used in |
| $dz_t$ | Shorthand for the gradient of the loss with respect to the pre-activation at step $t$ (before $\tanh$), computed as $dh_t \odot (1 - h_t^2)$ |
| Truncated BPTT | Splitting a long sequence into shorter chunks for backpropagation, carrying the hidden state but not the full gradient history across chunk boundaries |
| Unrolled computation graph | The view of an RNN's forward pass as a chain of identical cell computations across time steps, used to reason about both forward and backward passes |