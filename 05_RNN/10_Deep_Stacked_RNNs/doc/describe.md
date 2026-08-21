# 10. Deep Stacked RNNs

## Learning Objectives

- Explain how stacking RNN layers builds a feature hierarchy over time, analogous to depth in a CNN
- Implement a multi-layer RNN by feeding one layer's full output sequence as the next layer's input
- Understand the practical tradeoffs of adding depth to a recurrent network

## The Problem

Every architecture so far — plain RNN, LSTM, GRU, bidirectional — uses a single layer: one hidden state (or one pair of forward/backward hidden states) evolving through time. A single layer has real limits on what it can represent, just as a single convolutional layer does in the CNN module (Lesson 07 there: stacking conv layers builds a feature hierarchy from edges to parts to objects). Stacking RNN layers is the sequential analogue: build a hierarchy where each layer processes a progressively more abstract representation of the sequence, not just the raw input.

## The Concept

### Stacking: one layer's output sequence feeds the next layer's input

A single RNN layer transforms an input sequence into a hidden-state sequence. A **stacked** (deep) RNN feeds that entire output sequence as the *input* sequence to a second RNN layer, with its own independent set of weights, and so on for as many layers as desired:

```
Input sequence: x_1, x_2, ..., x_T
    |
    v
Layer 1 (RNN/LSTM/GRU):  produces h1_1, h1_2, ..., h1_T
    |
    v
Layer 2 (RNN/LSTM/GRU):  takes h1_1..h1_T AS ITS INPUT SEQUENCE
                          produces h2_1, h2_2, ..., h2_T
    |
    v
Layer 3: takes h2_1..h2_T as input, produces h3_1..h3_T
    |
    v
   ...
```

Each layer has entirely separate weights (its own `W_xh`, `W_hh`, `b_h`, or its own LSTM/GRU gate weights) — the layers are stacked in *depth*, not sharing parameters with each other, the same way stacked convolutional layers in a CNN each have their own filters.

```python
import torch.nn as nn

stacked_lstm = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    num_layers=3,          # stack 3 LSTM layers
    batch_first=True,
)
# output: the TOP layer's hidden state at every time step, shape (batch, T, hidden_size)
# h_n, c_n: final hidden/cell states from EVERY layer, shape (num_layers, batch, hidden_size)
```

### Why depth helps (the same logic as CNN depth, applied to time)

A single RNN layer at time `t` can only combine the current raw input with a summary of everything before it, through one nonlinear transformation per step. A second layer, operating on the *first layer's* hidden states rather than raw input, can learn to combine and abstract those first-layer patterns further — detecting relationships between the patterns the first layer found, rather than between raw inputs directly. This mirrors the CNN module's explanation of depth (Lesson 01/07 there): early layers learn simple, general patterns; later layers combine them into more complex, task-specific ones. For an RNN processing language, a rough (and inexact, but useful) intuition: lower layers might capture something closer to local syntax or word-level patterns, while higher layers capture longer-range or more semantic relationships — though unlike CNNs' well-studied edge-to-object hierarchy, what each RNN layer specifically learns is less crisply established and varies by task.

### Depth vs width vs simply widening a single layer

Adding layers (depth) and adding hidden units to a single layer (width) are both ways to increase a recurrent network's capacity, and they're not interchangeable: a single wide layer can combine raw input features in more ways at each individual time step, but only through **one** nonlinear transformation per step, whereas depth applies **multiple** nonlinear transformations in sequence at each time step, similar to why a deep feedforward network can represent functions a single very wide layer cannot easily approximate. In practice, 2-4 layers is a common range for stacked RNNs — deeper than that tends to run into vanishing/exploding gradient issues (Lesson 06) compounded across both time *and* depth simultaneously, along with diminishing returns and slower training, without the additional architectural aids (like residual/skip connections, covered for CNNs in that module's Lesson 15) that make very deep networks trainable elsewhere.

### Dropout between stacked layers

Since stacked RNNs add real capacity (and real overfitting risk, following the same logic as the CNN module's Lesson 10 on regularization), it's standard to apply dropout *between* layers — on the connections from one layer's output to the next layer's input, but typically *not* recurrently within a single layer's own time-step-to-time-step connections, since naively dropping units along the recurrent path can interfere with the network's ability to carry information across time at all (this specific concern, and a more careful "variational" form of recurrent dropout that avoids it, is covered in Lesson 16).

```python
stacked_lstm_with_dropout = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    num_layers=3,
    dropout=0.3,     # applied between layers, not within a layer's recurrence
    batch_first=True,
)
```

### Stacking combines naturally with bidirectionality

A stacked RNN can also be bidirectional at every layer (Lesson 09): each layer runs its own independent forward and backward pass, and the next layer's input is the concatenated forward+backward output of the previous layer. `nn.LSTM(num_layers=3, bidirectional=True, ...)` implements exactly this combination, and it's a common, strong default configuration for sequence-labeling tasks where the full sequence is available (per Lesson 09's usability constraint).

See `code/stacked_rnn_demo.py` for a from-scratch 2-layer stacked RNN (running Lesson 03's cell twice, layer 2 consuming layer 1's output sequence), verified against `torch.nn.RNN(num_layers=2)`, plus a demonstration of dropout's placement between layers.

## Exercises

1. Implement a 2-layer stacked RNN from scratch by running `rnn_cell_forward` once to produce a full hidden-state sequence, then running it again (with separate weights) treating that sequence as input. Verify against `torch.nn.RNN(num_layers=2)`.
2. Extend your implementation to 3 layers and confirm the final output shape and the shape of `h_n` (which should now have one hidden state per layer) match `torch.nn.RNN(num_layers=3)`'s conventions.
3. Train a 1-layer LSTM and a 3-layer LSTM (same hidden size) on the same task for the same number of epochs, and compare training loss curves — note whether the deeper network trains more slowly, per-epoch, and whether it reaches a better or worse final loss.
4. Combine stacking and bidirectionality (`num_layers=2, bidirectional=True`) and trace through the exact shape of the input each layer receives, given the concatenation from Lesson 09.

## Key Terms

| Term | What it actually means |
|---|---|
| Stacked (deep) RNN | An architecture where multiple RNN layers are chained, with each layer's full output sequence serving as the next layer's input sequence |
| num_layers | The parameter (in PyTorch's RNN/LSTM/GRU modules) controlling how many stacked layers are used |
| Inter-layer dropout | Dropout applied on the connections between stacked layers, as opposed to within a single layer's time-step-to-time-step recurrence |
