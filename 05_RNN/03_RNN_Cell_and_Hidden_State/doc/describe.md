# 03. RNN Cell and Hidden State

## Learning Objectives

- Write the exact equation an RNN cell uses to update its hidden state at each time step
- Implement a single RNN cell from scratch with NumPy and verify it against `torch.nn.RNNCell`
- Explain what each weight matrix in the cell is responsible for

## The Problem

Lesson 01 described the hidden state informally as "a summary of everything seen so far." This lesson makes that mechanical: exactly what arithmetic happens at every time step to produce the new hidden state from the current input and the previous hidden state — the single equation the entire RNN family is built from.

## The Concept

### The RNN cell equation

At every time step `t`, an RNN cell computes a new hidden state from the current input `x_t` and the previous hidden state `h_{t-1}`:

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
```

- `W_xh`: weight matrix mapping the input to the hidden state's space (shape: `hidden_size x input_size`)
- `W_hh`: weight matrix mapping the previous hidden state to the new hidden state (shape: `hidden_size x hidden_size`)
- `b_h`: a bias vector (shape: `hidden_size`)
- `tanh`: a nonlinearity (Lesson 06 of the CNN module covered activation functions generally; `tanh` — not ReLU — is the traditional default for RNN cells, discussed further in Lesson 06 of this module)

Every one of these — `W_xh`, `W_hh`, `b_h` — is the **same** at every time step: this is the weight-sharing-across-time idea from Lesson 01, made explicit. A sequence of length 50 doesn't need 50 different weight matrices; it applies this one equation 50 times, each time feeding the new hidden state back in as `h_{t-1}` for the next step.

### Reading the equation as "combine new information with memory"

`W_xh @ x_t` transforms the current input into the hidden state's space — "what does this new input contribute." `W_hh @ h_{t-1}` transforms the previous hidden state — "what do I already know, carried forward." Adding them together and squashing with `tanh` combines new information with existing memory into an updated summary. This is why the same weights, reused every step, can process sequences of any length: the cell doesn't need to "know" how long the sequence is: it just keeps combining the latest input with whatever it currently remembers.

### Producing an output

Often you also need an actual output at each step (a predicted next word, a per-step classification), not just the hidden state itself. This is a separate, additional transformation applied to the hidden state:

```
y_t = W_hy @ h_t + b_y
```

`W_hy` and `b_y` are yet another set of weights (shared across time steps, same as the others), mapping the hidden state to whatever output space the task needs (e.g. a probability distribution over a vocabulary, after a softmax). Not every task needs `y_t` at every step — Lesson 04 covers architectures that only read out `y_t` at the final step, or only at every step, depending on the task pattern from Lesson 01.

### Initializing the hidden state

Before the first real time step, `h_0` needs some starting value — conventionally a vector of zeros, representing "no information yet." `h_0` is occasionally itself a learned parameter (rather than fixed at zero) in some architectures, letting the network learn a useful default starting point, but zero-initialization is the standard default and what this module uses throughout.

### Implementing an RNN cell from scratch

```python
import numpy as np

def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    """x_t: (input_size,). h_prev: (hidden_size,). Returns new h_t: (hidden_size,)."""
    z = W_xh @ x_t + W_hh @ h_prev + b_h
    h_t = np.tanh(z)
    return h_t
```

This one function, called once per time step with the previous call's output fed back in as `h_prev`, is the entire mechanism Lesson 04 chains together into a full sequence forward pass.

See `code/rnn_cell_demo.py` for the complete implementation, verified numerically against `torch.nn.RNNCell` with identical weights, plus a demonstration of running the same cell repeatedly over several time steps to show the hidden state actually changing as new inputs arrive.

## Exercises

1. Implement `rnn_cell_forward` from scratch and verify it matches `torch.nn.RNNCell` exactly, given the same weights and the same input/hidden state.
2. By hand (or with the implementation), compute `h_1`, `h_2`, `h_3` for a toy 2-dimensional hidden state and a sequence of 3 simple inputs, starting from `h_0 = 0`. Confirm each hidden state differs from the last.
3. Set `W_hh` to a matrix of all zeros and observe what happens to the hidden state update. Explain in words what this configuration would mean for the network's ability to use "memory."
4. Set `W_xh` to a matrix of all zeros instead. Explain what the hidden state's evolution would look like in this case, and why the network could no longer respond to its input.

## Key Terms

| Term | What it actually means |
|---|---|
| RNN cell | The single reusable computational unit that updates a hidden state given the current input and the previous hidden state |
| W_xh | The weight matrix mapping the current input into the hidden state's space |
| W_hh | The weight matrix mapping the previous hidden state into the new hidden state, carrying memory forward |
| W_hy | The weight matrix mapping the hidden state to an output at a given time step |
| h_0 | The initial hidden state before the first real time step, conventionally a vector of zeros |
