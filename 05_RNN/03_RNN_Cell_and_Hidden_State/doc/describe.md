#  RNN Cell and Hidden State




##  The Problem: Making "Summary of Everything Seen So Far" Mechanical

The hidden state informally as "a summary of everything seen so far," and Lesson 02 sharpened this to "a fixed-size vector overwritten at every step." Neither, so far, tells you the actual arithmetic. This lesson closes that gap: exactly what computation happens at every time step to turn `x_t` and `h_{t-1}` into `h_t` — the single equation the entire RNN family (and, eventually, LSTM and GRU) is built from.

---

## The Concept: The RNN Cell Equation

### The equation itself

At every time step $t$, an RNN cell computes:

$$h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h)$$

| Symbol | Shape | Role |
|---|---|---|
| $x_t$ | $(d_x,)$ | current input vector |
| $h_{t-1}$ | $(d_h,)$ | previous hidden state |
| $W_{xh}$ | $(d_h, d_x)$ | maps the input into the hidden state's space |
| $W_{hh}$ | $(d_h, d_h)$ | maps the previous hidden state into the new hidden state |
| $b_h$ | $(d_h,)$ | bias vector |
| $\tanh$ | — | nonlinearity, squashes to $(-1, 1)$ |
| $h_t$ | $(d_h,)$ | new hidden state |

Here $d_x$ is the input dimension and $d_h$ is the hidden state dimension — these can be different sizes; $d_h$ is a design choice you make (bigger hidden state → more capacity to remember, but more parameters and compute).

**The crucial fact, restated from Lesson 01:** $W_{xh}$, $W_{hh}$, and $b_h$ are **the same at every time step**. A sequence of length 50 doesn't need 50 different weight matrices — it applies this one equation 50 times, each time feeding the freshly computed $h_t$ back in as $h_{t-1}$ for the next step.

###  Reading the equation as "combine new information with memory"

Split the equation into its two additive pieces and read each one on its own:

$$\underbrace{W_{xh}\, x_t}_{\text{"what does this new input contribute?"}} \;+\; \underbrace{W_{hh}\, h_{t-1}}_{\text{"what do I already know, carried forward?"}}$$

$W_{xh}x_t$ transforms the *current* input into the hidden state's coordinate system. $W_{hh}h_{t-1}$ transforms the *previous* hidden state — carrying forward whatever the network has already accumulated. Adding the two together, then squashing with $\tanh$, produces an updated summary that blends "what's new" with "what I remember." This is exactly why the same weights, reused every step, generalize to sequences of any length: the cell never needs to know how long the sequence is overall — it only ever needs to combine *this step's* input with *whatever it currently holds* in $h_{t-1}$.

###  A tiny worked example, step by step

Let's fix concrete numbers so you can watch the equation run. Suppose $d_x = 1$, $d_h = 2$ (a 2-dimensional hidden state, so you can see $W_{hh}$ actually be a real matrix, not just a scalar):

$$W_{xh} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}, \qquad W_{hh} = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}, \qquad b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \qquad h_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Input sequence: $x_1 = 1.0,\ x_2 = -1.0$.

**Step 1:**

$$W_{xh} x_1 + W_{hh} h_0 + b_h = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}(1.0) + \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}\begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}$$

$$h_1 = \tanh\left(\begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}\right) = \begin{bmatrix} 0.762 \\ 0.462 \end{bmatrix}$$

**Step 2:**

$$W_{xh} x_2 + W_{hh} h_1 + b_h = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}(-1.0) + \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}\begin{bmatrix} 0.762 \\ 0.462 \end{bmatrix} = \begin{bmatrix} -1.0 + 0.381 \\ -0.5 + 0.231 \end{bmatrix} = \begin{bmatrix} -0.619 \\ -0.269 \end{bmatrix}$$

$$h_2 = \tanh\left(\begin{bmatrix} -0.619 \\ -0.269 \end{bmatrix}\right) \approx \begin{bmatrix} -0.550 \\ -0.263 \end{bmatrix}$$

Notice both components of $h_2$ carry a trace of $h_1$ (via $W_{hh}h_1$) *and* respond to the new input $x_2$ (via $W_{xh}x_2$) — precisely the "combine new information with memory" reading , now visible in actual numbers. We'll verify these exact numbers against code .

### Producing an output

The hidden state $h_t$ isn't automatically an output — often you need an actual prediction at a given step (a predicted next word, a per-step classification), which is a **separate, additional transformation** applied on top of $h_t$:

$$y_t = W_{hy}\, h_t + b_y$$

$W_{hy}$ (shape $(d_y, d_h)$) and $b_y$ (shape $(d_y,)$) are yet another set of weights, shared across time steps just like the others, mapping the hidden state into whatever output space the task needs (e.g., a probability distribution over a vocabulary, after a softmax). Not every task needs $y_t$ at *every* step — Lesson 04 covers architectures that read out $y_t$ only at the final step, or at every step, depending on the input/output pattern introduced in Lesson 01 .

###  Initializing the hidden state

Before the first real time step, $h_0$ needs *some* starting value. The conventional choice is a vector of zeros — representing "no information yet." In some architectures, $h_0$ is itself a learned parameter rather than fixed at zero, letting the network learn a useful default starting point instead of always starting from a blank slate. Zero-initialization is the standard default, and what this module uses throughout unless stated otherwise.

###  Two degenerate cases worth understanding

Setting a weight matrix to zero is a good diagnostic tool for understanding what it's *for*, because it isolates that term's contribution:

- **If $W_{hh} = 0$:** the equation collapses to $h_t = \tanh(W_{xh}x_t + b_h)$ — $h_{t-1}$ no longer contributes at all. Each hidden state would depend *only* on the current input, with zero memory of anything before it. This is exactly a plain feedforward transformation applied independently at each time step; the "recurrent" part of the RNN would be gone entirely.
- **If $W_{xh} = 0$:** the equation collapses to $h_t = \tanh(W_{hh}h_{t-1} + b_h)$ — $x_t$ no longer contributes at all. The hidden state would keep evolving step by step, but completely obliviously to the actual input sequence; it would produce the exact same trajectory of hidden states regardless of what data you fed it. The network would be "remembering" something, but never anything about the real world it's supposed to be processing.

A working RNN cell needs both terms: $W_{xh}$ to stay responsive to new input, $W_{hh}$ to retain a trace of the past. Exercises 3–4 ask you to confirm this yourself, numerically.

---

##  Reading the Equation as Code

```python
import numpy as np

def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    """
    x_t:    (input_size,)
    h_prev: (hidden_size,)
    W_xh:   (hidden_size, input_size)
    W_hh:   (hidden_size, hidden_size)
    b_h:    (hidden_size,)
    Returns h_t: (hidden_size,)
    """
    z = W_xh @ x_t + W_hh @ h_prev + b_h
    h_t = np.tanh(z)
    return h_t
```

This one function, called once per time step with the previous call's output fed back in as `h_prev`, *is* the entire mechanism — Lesson 04 chains it together into a full sequence forward pass, and nothing new is added to the cell itself.

---

##  Use It: Verifying the Worked Example in Code

Let's check  hand-worked numbers against this implementation, and then cross-check the whole thing against PyTorch's built-in `RNNCell` to make sure our from-scratch version is faithful to the standard definition.

###  Reproducing the by-hand numbers

```python
import numpy as np

W_xh = np.array([[1.0], [0.5]])                 # shape (2, 1)
W_hh = np.array([[0.5, 0.0], [0.0, 0.5]])        # shape (2, 2)
b_h  = np.array([0.0, 0.0])                      # shape (2,)

def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    z = W_xh @ x_t + W_hh @ h_prev + b_h
    return np.tanh(z)

h = np.array([0.0, 0.0])                # h_0
inputs = [np.array([1.0]), np.array([-1.0])]   # x_1, x_2

for t, x_t in enumerate(inputs, start=1):
    h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
    print(f"h_{t} = {h}")
```

```
h_1 = [0.76159416 0.46211716]
h_2 = [-0.5500522  -0.26289402]
```

These match  (rounding differences aside) — a good sign our implementation is faithful to the equation.

###  Cross-checking against `torch.nn.RNNCell`

```python
import torch
import torch.nn as nn

cell = nn.RNNCell(input_size=1, hidden_size=2, nonlinearity="tanh")

# Load the exact same weights so the two implementations are directly comparable.
with torch.no_grad():
    cell.weight_ih.copy_(torch.tensor(W_xh, dtype=torch.float32))
    cell.weight_hh.copy_(torch.tensor(W_hh, dtype=torch.float32))
    cell.bias_ih.zero_()
    cell.bias_hh.zero_()

h_torch = torch.zeros(1, 2)   # batch dimension of 1
for t, x_val in enumerate([1.0, -1.0], start=1):
    x_t = torch.tensor([[x_val]], dtype=torch.float32)
    h_torch = cell(x_t, h_torch)
    print(f"h_{t} (torch) = {h_torch.detach().numpy()}")
```

Running this should reproduce the same `h_1` and `h_2` as , up to floating-point rounding — confirming that our from-scratch `rnn_cell_forward` implements exactly the same equation PyTorch's built-in cell does. (Note PyTorch splits the bias into `bias_ih` and `bias_hh`, which together equal our single $b_h$; zeroing both keeps this example matching , which used $b_h = 0$.)

---

## Exercises

**Working the equation by hand**

1. Using $W_{xh} = \begin{bmatrix}1.0\\0.5\end{bmatrix}$, $W_{hh} = \begin{bmatrix}0.5 & 0.0\\0.0 & 0.5\end{bmatrix}$, $b_h = \begin{bmatrix}0\\0\end{bmatrix}$ , compute $h_3$ by hand given $x_3 = 2.0$, continuing from the $h_2$ found . Show your work.
2. For a hidden state of dimension $d_h = 2$ and an input of dimension $d_x = 3$, write out the exact shapes of $W_{xh}$, $W_{hh}$, and $b_h$. Then do the same for $d_h = 128$, $d_x = 300$ (a realistic word-embedding size) — how many total learned parameters does the cell have in this second case? (Don't forget $b_h$.)

**Understanding the two weight matrices**

3. Set `W_hh` to a matrix of all zeros in the code , and rerun it for the same two inputs. Confirm the resulting `h_1` and `h_2` match what predicts (that $h_{t-1}$ no longer contributes). Then explain in your own words what this configuration would mean for the network's ability to use "memory" across a long sequence.
4. Set `W_xh` to a matrix of all zeros instead (keep `W_hh` ), and rerun the code for the same two inputs. Confirm the hidden state evolves identically regardless of what you set `inputs` to. Explain why this happens, referencing .

**Extending the implementation**

5. Extend `rnn_cell_forward`  to also compute the output $y_t = W_{hy}h_t + b_y$ , given additional parameters `W_hy` and `b_y`. Test it on the sequence , using an output dimension of $d_y = 1$.
6. (Open-ended) The `torch.nn.RNNCell` cross-check  initializes `h_torch` to zeros directly. Modify that code to instead treat $h_0$ as a *learned* parameter (a `torch.nn.Parameter` of shape `(1, 2)`) as mentioned , and confirm gradients flow back to it after a backward pass on some toy loss.

---

## Key Terms

| Term | What it actually means |
|---|---|
| RNN cell | The single reusable computational unit that updates a hidden state given the current input and the previous hidden state, via $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ |
| $W_{xh}$ | The weight matrix mapping the current input into the hidden state's space — responsible for responsiveness to new input |
| $W_{hh}$ | The weight matrix mapping the previous hidden state into the new hidden state — responsible for carrying memory forward |
| $W_{hy}$ | The weight matrix mapping the hidden state to an output at a given time step |
| $b_h$, $b_y$ | Bias vectors added in the hidden-state update and the output computation, respectively |
| $h_0$ | The initial hidden state before the first real time step, conventionally a vector of zeros (sometimes a learned parameter instead) |