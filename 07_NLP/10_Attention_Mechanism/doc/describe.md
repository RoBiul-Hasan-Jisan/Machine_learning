# 10. Attention Mechanism — The Breakthrough

## Learning Objectives

- Explain how attention removes the fixed-context-vector bottleneck from Lesson 09
- Implement additive (Bahdanau) attention from scratch within a seq2seq decoder
- Interpret attention weights as a soft alignment between output and input positions

## The Problem

Lesson 09 ended on a concrete limitation: the encoder-decoder architecture forces the *entire* input sequence through one fixed-size context vector, and performance measurably degrades as input length grows — exactly what the lesson's bottleneck experiment showed empirically. Attention (Bahdanau et al., 2014; refined by Luong et al., 2015) fixes this with a deceptively simple idea: instead of forcing the decoder to work from one compressed summary, let it look back at *every* encoder hidden state at every decoding step, and learn which ones matter most for the token it's currently generating.

## The Concept

### The core idea: a learned, per-step weighted average of encoder states

Instead of passing only the encoder's *final* hidden state to the decoder (Lesson 09's bottleneck), attention keeps *all* the encoder's hidden states around, and at each decoder step, computes a fresh weighted combination of them — different weights at every decoding step, depending on what the decoder currently needs.

```
Encoder hidden states (kept, not discarded):  h_1, h_2, h_3, ..., h_T

At decoder step t, generating one output token:
  1. Compute an "attention score" between the decoder's current state and EVERY encoder state
  2. Softmax those scores into attention WEIGHTS (summing to 1)
  3. Take the weighted sum of encoder states, using those weights, as this step's context vector

context_t = sum over i of: attention_weight(t, i) * h_i
```

The critical difference from Lesson 09: `context_t` is recomputed fresh at *every* decoder step `t`, and can weight completely different encoder positions depending on what's being generated — the decoder isn't stuck with one static summary of the whole input; it gets to "look back" and focus wherever is currently relevant.

### Computing attention scores: additive (Bahdanau) attention

The original formulation scores compatibility between the decoder's current hidden state and each encoder hidden state using a small feedforward network:

```
score(s_t, h_i) = v^T . tanh(W_s @ s_t + W_h @ h_i)

s_t   = decoder's current hidden state
h_i   = encoder's hidden state at position i
W_s, W_h, v = learned weight matrices/vector

attention_weights = softmax(scores across all i)     <- one weight per encoder position, summing to 1
context_t = sum_i(attention_weights[i] * h_i)          <- the weighted combination
```

This small feedforward scoring network is trained jointly with the rest of the model — nothing about "which positions matter" is hand-specified; the network learns it purely by whatever minimizes the overall sequence-to-sequence loss.

### Attention weights as a soft alignment

Because attention weights sum to 1 across encoder positions, they form a probability-like distribution over the input — visualized as a matrix (decoder steps × encoder positions), this reveals which input words the model is "looking at" when generating each output word:

```
              how   are   you
comment       0.85  0.10  0.05
allez         0.05  0.80  0.15
vous          0.05  0.15  0.80
?             0.60  0.20  0.20
```

This isn't a hard, discrete alignment (attention weights are continuous, and typically spread some weight across multiple positions) but it's directly interpretable — and in machine translation specifically (Lesson 11), attention weight matrices closely resemble the word-alignment tables that classical statistical translation systems computed explicitly, except here they emerge automatically as a byproduct of training, without anyone specifying alignment rules.

### Why this removes the bottleneck

The decoder no longer depends on cramming the entire input into one fixed-size vector before generation even starts — every encoder position remains directly, individually accessible at every decoding step, through the attention mechanism's weighted sum. Information from the first word of a long input is exactly as directly reachable as information from the last word, at any decoding step — unlike Lesson 09's plain encoder-decoder, where early information had to survive being carried forward through every subsequent encoding step before reaching the final context vector.

### Attention inside a decoder step

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_state, encoder_states):
        # decoder_state: (batch, hidden_size)
        # encoder_states: (batch, T, hidden_size)
        decoder_expanded = decoder_state.unsqueeze(1)               # (batch, 1, hidden_size)
        scores = self.v(torch.tanh(self.W_s(decoder_expanded) + self.W_h(encoder_states)))
        scores = scores.squeeze(-1)                                   # (batch, T)
        weights = F.softmax(scores, dim=-1)                           # (batch, T)
        context = torch.bmm(weights.unsqueeze(1), encoder_states)     # (batch, 1, hidden_size)
        return context.squeeze(1), weights
```

At every decoder time step, this module is called fresh: it takes the decoder's current hidden state and *all* encoder hidden states, and returns both a context vector (used alongside the decoder's own input to produce the next hidden state and output) and the attention weights themselves (useful for interpretation/visualization, as shown above).

### Where this leads

This lesson's attention mechanism is specifically the RNN-based encoder-decoder attention originally introduced for machine translation (Lesson 11 covers that application directly). The same core idea — a learned, weighted combination over a set of representations, computed via query/key-style compatibility scoring — generalizes far beyond this one architecture: it's the direct conceptual ancestor of Transformer self-attention (covered in the RNN module's Lesson 18), where the same query/compatibility/weighted-sum pattern is applied *within* a single sequence rather than between an encoder and decoder, and removes recurrence from the picture entirely.

See `code/attention_demo.py` for a complete additive-attention seq2seq model trained on the same reversal task from Lesson 09, with a direct accuracy comparison against Lesson 09's plain encoder-decoder as sequence length grows, plus a visualization of the learned attention weight matrix.

## Exercises

1. Implement `AdditiveAttention` and integrate it into Lesson 09's decoder (feeding the context vector alongside the decoder's own input at each step). Confirm the model still trains successfully.
2. Repeat Lesson 09's bottleneck experiment (accuracy vs sequence length) with the attention-augmented model, and compare the degradation curve against the plain encoder-decoder's.
3. Extract and visualize (as a heatmap or printed matrix) the attention weights for a few example predictions. Check whether the weights concentrate near the "expected" aligned position for a task like sequence reversal, where the correct alignment is known in advance.
4. Implement multiplicative (Luong-style) attention, `score(s_t, h_i) = s_t^T @ W @ h_i`, and compare its behavior and training speed against additive attention on the same task.

## Key Terms

| Term | What it actually means |
|---|---|
| Attention mechanism | A technique that computes a fresh, learned weighted combination of encoder states at every decoder step, removing the fixed-context bottleneck |
| Attention score | A learned compatibility measure between a decoder state and an encoder state, before normalization into weights |
| Attention weights | The softmax-normalized attention scores across all encoder positions, forming a distribution that sums to 1 |
| Additive (Bahdanau) attention | An attention scoring function using a small feedforward network with a tanh nonlinearity to compute compatibility |
| Soft alignment | The interpretation of attention weights as a continuous, learned mapping between output positions and relevant input positions |
