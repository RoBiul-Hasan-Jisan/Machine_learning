# 11. Sequence-to-Sequence Models

## Learning Objectives

- Explain why translation-style tasks need a different architecture than the many-to-many pattern from Lesson 04
- Implement an encoder-decoder architecture with a context vector
- Identify the information bottleneck this design creates, motivating attention (Lesson 12)

## The Problem

Lesson 04's many-to-many pattern reads an output at every input time step — useful for tasks like part-of-speech tagging, where input and output are the same length and aligned position-by-position. Translation isn't like that: "I am a student" (4 words) might translate to "Je suis étudiant" (3 words) in French, or to a 6-word sentence in another language. There's no natural one-to-one alignment between input and output positions, and the output length isn't even known in advance. This needs a genuinely different architecture.

## The Concept

### Encoder-decoder: two separate RNNs, connected by a context vector

A sequence-to-sequence (seq2seq) model uses two separate RNNs (each can be a plain RNN, LSTM, or GRU, and each can be stacked or bidirectional per Lessons 09-10): an **encoder** that reads the entire input sequence and compresses it into a single fixed-size vector, and a **decoder** that generates the output sequence from that vector, one token at a time.

```
ENCODER (reads the input, produces one summary vector):

x_1 -> [RNN] -> x_2 -> [RNN] -> x_3 -> [RNN] -> x_4 -> [RNN]
                                                    |
                                                    v
                                            context vector c
                                        (the encoder's FINAL hidden state)

DECODER (generates the output, one token at a time, starting from c):

c -> [RNN] -> y_1 -> [RNN] -> y_2 -> [RNN] -> y_3 -> <END>
       ^ initial hidden state = c
```

The **context vector** `c` is simply the encoder's final hidden state (or, for an LSTM encoder, both the final hidden and cell state) — a fixed-size summary of the entire input sequence, regardless of how long that input was. This is the key structural idea that solves the "different lengths, no alignment" problem: the decoder never sees the encoder's individual per-step hidden states directly, only this one compressed summary, so nothing about the decoder's architecture needs to know or care how long the input sequence was.

### The decoder: one-to-many, driven by its own previous output

The decoder follows Lesson 04's one-to-many pattern: at each step, it takes its own previous output (fed back in as input) plus its running hidden state, and produces the next output token, continuing until it generates a special `<END>` token or hits a maximum length.

```python
def decoder_forward(context_vector, decoder_cell, output_layer, start_token, end_token, max_len=50):
    h = context_vector
    token = start_token
    outputs = []
    for _ in range(max_len):
        h = decoder_cell(token, h)
        token = output_layer(h)          # e.g. argmax over a vocabulary distribution
        if token == end_token:
            break
        outputs.append(token)
    return outputs
```

### Training: teacher forcing

During training, feeding the decoder's own (possibly wrong, especially early in training) predictions back in as input for the next step can compound errors and slow learning. **Teacher forcing** instead feeds the *true* previous target token as input during training, regardless of what the decoder actually predicted:

```
Without teacher forcing (used at inference, when there's no ground truth):
  decoder's OWN prediction at step t  ->  fed in as input at step t+1

With teacher forcing (used during training):
  the TRUE target token at step t     ->  fed in as input at step t+1
  (even if the decoder's own prediction at step t was wrong)
```

This tends to speed up and stabilize training, at the cost of a training/inference mismatch: the decoder is trained to always predict the next token given the *correct* preceding context, but at inference time it must condition on its own, potentially imperfect, past predictions — a discrepancy called **exposure bias**. A common practical compromise, scheduled sampling, gradually shifts from teacher forcing toward using the model's own predictions as training progresses, narrowing that mismatch.

### The bottleneck problem

The entire input sequence — however long — gets compressed into one fixed-size context vector, and the decoder has to reconstruct the whole output from that single vector alone. For short sequences this works reasonably well; for long sequences (a long sentence, a whole paragraph), a single fixed-size vector is an increasingly severe information bottleneck — there's only so much a fixed number of floating-point values can encode, no matter how long the original sequence was. Empirically, plain encoder-decoder seq2seq models' translation quality degrades noticeably as input sentence length grows, tracing directly back to this bottleneck.

This limitation is exactly what motivates Lesson 12's attention mechanism: instead of forcing the decoder to work from one fixed summary vector, attention lets the decoder look back at *all* of the encoder's per-step hidden states directly, choosing which parts of the input are relevant at each decoding step — removing the fixed-size bottleneck rather than working around it.

### Implementing a basic seq2seq model

```python
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, source_seq, target_seq):
        # Encoder: read the full input, keep only the final (h, c)
        source_embedded = self.embedding(source_seq)
        _, (h, c) = self.encoder(source_embedded)

        # Decoder: initialized from the encoder's final state, teacher-forced with target_seq
        target_embedded = self.embedding(target_seq)
        decoder_output, _ = self.decoder(target_embedded, (h, c))
        return self.output_layer(decoder_output)
```

See `code/seq2seq_demo.py` for a complete, runnable seq2seq model trained on a small synthetic "reverse the sequence" task (input and output lengths deliberately differ from a real alignment, to exercise the encoder-decoder pattern), with teacher forcing during training and free-running (no teacher forcing) generation at inference time.

## Exercises

1. Implement the `Seq2Seq` class above and train it on a toy task where the target is the input sequence reversed (e.g. `[3, 7, 1] -> [1, 7, 3]`). Confirm it learns to reverse short sequences correctly.
2. Modify training to use teacher forcing only 50% of the time (randomly choosing, at each decoder step, between the true previous target and the model's own prediction) and compare convergence speed against always using teacher forcing.
3. Train the same architecture on the reversal task with increasing input lengths (5, 10, 20, 40 tokens) and measure accuracy at each length, to observe the bottleneck problem directly.
4. Implement free-running generation (no teacher forcing, feeding the model's own prediction back in) for inference, including stopping at a generated `<END>` token, and compare generated outputs against teacher-forced training-time predictions on the same input.

## Key Terms

| Term | What it actually means |
|---|---|
| Sequence-to-sequence (seq2seq) | An architecture pattern using an encoder RNN and a decoder RNN to map an input sequence to an output sequence of a possibly different length |
| Encoder | The RNN that reads the entire input sequence and compresses it into a context vector |
| Decoder | The RNN that generates the output sequence, one token at a time, starting from the encoder's context vector |
| Context vector | The encoder's final hidden state (and cell state, for LSTM), serving as a fixed-size summary of the entire input sequence |
| Teacher forcing | Feeding the true target token (rather than the model's own prediction) as the decoder's input during training |
| Exposure bias | The training/inference mismatch created by teacher forcing, since at inference the decoder must condition on its own (possibly imperfect) past predictions |
