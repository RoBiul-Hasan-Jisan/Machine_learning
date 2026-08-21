# 01. Introduction to RNN

## Learning Objectives

- Explain why feedforward networks and CNNs struggle with sequential data
- Describe the core idea of a Recurrent Neural Network: a hidden state carried across time steps
- Recognize the range of tasks RNNs are suited to and where this module is heading

## The Problem

A CNN (covered in this curriculum's CNN module) is built around a fixed, local, spatial structure: a filter slides across a grid of pixels, and every position is treated the same way regardless of order. A fully connected network needs a fixed-size input, with every input value in a fixed, meaningful position. Neither assumption holds for sequential data — text, audio, time series, DNA sequences — where:

- **Length varies.** A sentence can be 3 words or 300. A fixed-size input can't accommodate both without padding or truncation that throws away information or wastes computation.
- **Order matters.** "The dog bit the man" and "The man bit the dog" contain the same words but mean opposite things. A model that ignores order (like a plain bag-of-words feature vector) can't distinguish them.
- **Context accumulates.** Understanding word 50 in a sentence often depends on words 1 through 49. A network needs some notion of *memory* — carrying forward what it has seen so far — that neither a CNN's local receptive field nor a plain feedforward layer provides.

## The Concept

### The core idea: a hidden state carried through time

A Recurrent Neural Network processes a sequence one element at a time, maintaining a **hidden state** — a vector summarizing everything relevant seen so far — that gets updated at every step and fed forward into the next one.

```
x_1        x_2        x_3        x_4
 |          |          |          |
 v          v          v          v
[RNN] -h1-> [RNN] -h2-> [RNN] -h3-> [RNN] -h4->
 |          |          |          |
 v          v          v          v
y_1        y_2        y_3        y_4

Same RNN cell (same weights) reused at every time step.
h_t depends on x_t AND h_(t-1) -- the network's "memory" of everything before it.
```

This is the sequential analogue of a CNN's weight sharing (covered in the CNN module's Lesson 01): instead of reusing the same filter across spatial positions, an RNN reuses the same weights across *time steps*. This is what lets an RNN handle sequences of any length with a fixed number of parameters — the same small set of weights gets applied once per time step, however many time steps there are.

### What makes this different from a feedforward network

A feedforward network computes `output = f(input)` — one pass, no memory of anything outside the current input. An RNN computes `hidden_t = f(input_t, hidden_{t-1})` — every step's output depends not just on the current input, but on a running summary of every input that came before it. This recurrence (the hidden state feeding back into the next step's computation) is where the "recurrent" in the name comes from, and it's the single idea this whole module builds outward from.

### The range of tasks RNNs handle

```
One-to-many:    single input -> sequence output       (e.g. image captioning)
Many-to-one:    sequence input -> single output        (e.g. sentiment classification)
Many-to-many:   sequence input -> sequence output       (e.g. translation, per-word tagging)
```

The same core mechanism — a hidden state updated one step at a time — supports all three patterns; what differs is where inputs are fed in and where outputs are read out, a distinction Lesson 04 makes concrete.

### Where this module goes from here

| Lessons | What they cover |
|---|---|
| 02-04 | Representing sequential data, the RNN cell's mechanics, and a full forward pass |
| 05-06 | Backpropagation through time, and the specific gradient problems that come with it |
| 07-09 | LSTM, GRU, and bidirectional RNNs — architectures designed to fix those gradient problems and use context from both directions |
| 10-12 | Stacking layers, sequence-to-sequence models, and attention — building toward more capable architectures |
| 13-16 | Practical techniques: word embeddings, text generation, time series forecasting, regularization |
| 17-18 | Training RNNs in practice, and how RNNs compare to the Transformer architecture that has largely succeeded them |
| 19 | End-to-end projects combining everything |

### Why RNNs, in the era of Transformers

Transformers (the architecture behind most modern large language models) have replaced RNNs for most large-scale text tasks, primarily because RNNs process a sequence strictly one step at a time — step `t` can't start until step `t-1` finishes — which makes them hard to parallelize across a long sequence during training. Lesson 18 covers this comparison directly. Even so, RNNs (and specifically their LSTM/GRU variants) remain genuinely useful: they're simpler to reason about, cheaper for smaller-scale or streaming/online tasks where data arrives one step at a time, still a strong default for many time-series problems, and — importantly for this module — they're the conceptual foundation that attention and Transformers were built in response to. Understanding what RNNs do well and where they struggle is what makes the Transformer's design choices make sense, rather than feeling arbitrary.

## Exercises

1. List three real-world tasks that are naturally sequential (order-dependent) and explain, for each, what "memory" of earlier steps the task requires.
2. Take the sentence "The dog bit the man" and its reordering "The man bit the dog." Explain concretely why a bag-of-words feature vector (word counts, ignoring order) cannot distinguish them, while a sequential model in principle can.
3. For each of one-to-many, many-to-one, and many-to-many, name one additional real-world task beyond the examples given above.

## Key Terms

| Term | What it actually means |
|---|---|
| Recurrent Neural Network (RNN) | A neural network architecture that processes a sequence one element at a time, maintaining a hidden state updated at every step |
| Hidden state | A vector summarizing everything the network has processed so far in a sequence, carried forward and updated at each time step |
| Sequential data | Data where order carries meaning and length can vary, such as text, audio, or time series |
| Recurrence | The property that a step's computation depends on the result of the previous step, creating a feedback loop through time |
