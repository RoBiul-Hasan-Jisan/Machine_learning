# 09. Bidirectional RNN

## Learning Objectives

- Explain why a standard (unidirectional) RNN only has access to past context, and when that's a limitation
- Implement a bidirectional RNN by combining a forward and a backward pass
- Identify which tasks benefit from bidirectionality and which cannot use it

## The Problem

Every architecture so far — plain RNN (Lesson 03), LSTM (Lesson 07), GRU (Lesson 08) — processes a sequence strictly left to right: the hidden state at step `t` depends only on `x_1, ..., x_t`, never on anything later in the sequence. For many tasks this is exactly right (you can't use tomorrow's stock price to predict today's). But for tasks where the *entire* sequence is available at once before you need to make any prediction — a sentence you're translating, a full time series you're analyzing after the fact, an audio clip you're transcribing — throwing away the information in the *future* part of the sequence when interpreting an earlier part is a real, avoidable loss.

## The Concept

### A concrete illustration of the limitation

```
"The bank was steep, so the hikers turned back."
"The bank was closed, so I couldn't withdraw cash."
```

Interpreting the word "bank" correctly requires reading past it — "steep" or "closed" disambiguates the meaning, but a strictly left-to-right RNN, at the moment it processes "bank," has no access to that disambiguating word yet. If the task is to classify each word's sense (river bank vs financial bank) *after* seeing the entire sentence, a unidirectional RNN is unnecessarily handicapped: the information needed to disambiguate is right there in the input, just later in the sequence.

### The fix: run two RNNs, one in each direction

A bidirectional RNN runs two entirely separate RNNs (or LSTMs, or GRUs — bidirectionality is a wrapper that works with any of them) over the same sequence: one processing it left-to-right as usual, and a second processing it right-to-left, then combines their hidden states at each position.

```
Forward RNN:   x_1 -> h_1_fwd -> x_2 -> h_2_fwd -> x_3 -> h_3_fwd -> ...
Backward RNN:  x_T -> h_T_bwd -> x_(T-1) -> h_(T-1)_bwd -> ... -> x_1 -> h_1_bwd

At each position t, combine both:   h_t = [h_t_fwd ; h_t_bwd]     (concatenation)
```

The forward RNN's hidden state at position `t` summarizes everything *before and including* `t`. The backward RNN's hidden state at position `t` (running right-to-left) summarizes everything *after and including* `t`. Concatenating them gives, at every position, a representation informed by the *entire* sequence — both what came before and what comes after — solving exactly the "bank" disambiguation problem above.

```python
import torch.nn as nn

bidirectional_lstm = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    bidirectional=True,     # runs a second, backward-direction LSTM internally
    batch_first=True,
)
# output shape: (batch, T, 2 * hidden_size)  -- forward and backward concatenated
```

The two directions have **entirely separate weights** — the forward RNN and backward RNN don't share any parameters, they're two independent networks whose outputs happen to get combined. This roughly doubles the parameter count (and compute) relative to a unidirectional RNN of the same hidden size, which is the direct cost of the improved context.

### When bidirectionality is (and isn't) usable

The critical requirement: **the entire sequence must be available before processing begins.** This makes bidirectional RNNs a natural fit for tasks like:

- Sentence classification (the full sentence exists before you classify it)
- Named entity recognition, part-of-speech tagging (label every word, using the full sentence as context)
- Speech-to-text on a pre-recorded audio clip

And a poor fit — in fact, unusable — for genuinely **online/streaming/real-time** tasks, where predictions must be made as data arrives, before the "future" part of the sequence even exists yet:

- Real-time speech transcription, where you need a transcription as someone speaks, not after they finish
- Next-word prediction while someone is actively typing
- Any live/streaming time series forecast, where "the future" is literally the thing being predicted

This is a hard constraint, not a design tradeoff to be tuned: a backward pass over data that hasn't happened yet is not a slower or more expensive option, it's simply not possible. Confirming which category a task falls into is the first thing to check before reaching for a bidirectional architecture.

### Combining forward and backward states: concatenation vs other options

Concatenation (shown above) is the standard combination method, preserving both directions' information separately for downstream layers to use as they see fit. Alternatives (summing the two hidden states, or averaging them) compress more aggressively and are used occasionally, but concatenation is the default in essentially all standard library implementations (`nn.LSTM(bidirectional=True)`, `nn.GRU(bidirectional=True)`) and this module follows that convention throughout.

### Stacking with bidirectionality

Bidirectional RNNs combine naturally with the stacked/deep RNN idea (Lesson 10): each layer can itself be bidirectional, with the next layer's input being the concatenated forward+backward output of the previous layer. This is the standard setup for strong sequence-labeling and encoder architectures before attention-based methods (Lesson 12) became dominant.

See `code/bidirectional_demo.py` for a from-scratch bidirectional RNN (running two independent `rnn_cell_forward` passes and concatenating), verified against `torch.nn.RNN(bidirectional=True)`, plus a demonstration that the backward direction's hidden state at position `t` genuinely depends on later time steps.

## Exercises

1. Implement a bidirectional RNN from scratch by running the Lesson 03 `rnn_cell_forward` once forward and once backward over a sequence, then concatenating the two hidden state sequences. Verify against `torch.nn.RNN(bidirectional=True)`.
2. Confirm empirically that the backward direction's hidden state at position 1 changes if you change the value of `x_T` (the last input), while a plain forward RNN's hidden state at position 1 does not.
3. For a named entity recognition task (tagging each word as person/place/organization/none), explain concretely why a bidirectional RNN would likely outperform a unidirectional one, using a specific example sentence.
4. Take a real-time transcription task and explain in writing why bidirectionality cannot be applied directly, and what a practical alternative might be (hint: consider processing in small, deliberately delayed chunks rather than one token in true real-time).

## Key Terms

| Term | What it actually means |
|---|---|
| Bidirectional RNN | An architecture combining a forward-direction RNN and an independent backward-direction RNN, giving every position access to both past and future context |
| Unidirectional RNN | A standard RNN processing a sequence in one direction only, with each position's hidden state depending solely on earlier positions |
| Online / streaming task | A task where predictions must be made as data arrives, before later parts of the sequence exist, ruling out bidirectionality |
