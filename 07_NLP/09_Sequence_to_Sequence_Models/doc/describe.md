# 09. Sequence-to-Sequence Models

## Learning Objectives

- Explain the encoder-decoder architecture and why it handles input/output sequences of different lengths
- Implement a basic encoder-decoder model from scratch and train it on a synthetic mapping task
- Identify the fixed-context-vector bottleneck that motivates attention (Lesson 10)

## The Problem

Many NLP tasks map one sequence to a *different* sequence — not the same length, not even the same vocabulary in translation's case. "How are you?" (3 words) might translate to "Comment allez-vous?" (3 words in French, but a genuinely different mapping) or to a language where the natural translation is 6 words. None of the many-to-one or many-to-many-same-length patterns covered so far (Lesson 04 of the RNN module) handle this: they either produce one output, or exactly as many outputs as inputs. Sequence-to-sequence (seq2seq) models are built specifically for input and output sequences of independent, differing lengths.

## The Concept

### The encoder-decoder architecture

A seq2seq model splits the problem into two RNNs (or LSTM/GRUs) with different jobs: an **encoder** reads the entire input sequence and compresses it into a fixed-size vector, and a **decoder** generates the output sequence, one token at a time, conditioned on that vector.

```
Input:  "how" "are" "you"
           |     |     |
        [Encoder RNN, processes the whole input]
           |     |     |
          h1    h2    h3  <- final hidden state h3 becomes the "context vector"
                        |
                        v
              context vector (fixed size, summarizes the ENTIRE input)
                        |
                        v
        [Decoder RNN, generates output one token at a time]
           |     |     |      |
        "comment" "allez" "vous" "?"
```

The encoder is a standard many-to-one RNN (RNN module Lesson 04): it consumes the whole input and its *final* hidden state is treated as a compressed summary — the context vector — of everything in the input sequence. The decoder is then a one-to-many RNN (same lesson): starting from that context vector as its initial hidden state, it generates output tokens one at a time, typically feeding each generated token back in as the next step's input (the RNN module's Lesson 04 also covered this feedback pattern, and it reappears centrally in Lesson 16's text generation).

### Training: teacher forcing

During training, the true target sequence is known in advance, so rather than feeding the decoder's own (possibly wrong, especially early in training) predictions back in as input, **teacher forcing** feeds the *true* previous target token as input at each decoder step instead:

```
Decoder training, target = "comment allez vous ?"

Step 1: input = <START>,     true target = "comment"   (loss computed against "comment")
Step 2: input = "comment",   true target = "allez"      (the TRUE previous word, not the model's own guess)
Step 3: input = "allez",     true target = "vous"
Step 4: input = "vous",      true target = "?"
```

This stabilizes and speeds up training substantially, since the decoder always sees a "correct" history to condition on rather than potentially compounding its own early mistakes. The tradeoff: at *inference* time, there's no true target sequence available, so the decoder must feed back its own predictions (exactly the one-to-many generation pattern) — creating a training/inference mismatch (sometimes called "exposure bias") where the model was never trained on the kind of imperfect, self-generated history it will actually see at deployment. Lesson 16 covers decoding strategies that partly mitigate this.

### The end-of-sequence token

Since output length isn't fixed or known in advance, the decoder needs a way to signal "I'm done generating." A special `<END>` (or `<EOS>`) token, included in training targets, lets the decoder learn to predict when to stop — at inference, generation continues until the decoder produces this token (or a maximum length is reached, as a safety cutoff).

### Implementing a basic encoder-decoder from scratch

```python
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, h_n = self.rnn(embedded)
        return h_n   # the context vector: (1, batch, hidden_size)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
```

The encoder's final hidden state (`h_n`) is passed directly as the decoder's *initial* hidden state — this single vector is the entire channel through which the decoder learns anything about the input sequence.

### The fixed-context-vector bottleneck

This architecture has a real, structural limitation: no matter how long the input sequence is — 5 words or 50 — it gets compressed into the *same fixed-size* context vector. For short sequences this is a manageable compression; for long sequences, the encoder is forced to cram an increasing amount of information into a vector of unchanging size, and in practice performance degrades noticeably as input length grows, since early information gets progressively diluted or overwritten by the time the encoder finishes processing a long sequence (a specific instance of the RNN module's Lesson 06 vanishing-gradient/limited-memory concern, here affecting what the encoder can retain by its final step, not just what gradients can reach during training).

This bottleneck — one fixed-size vector forced to represent an entire, arbitrarily long input — is precisely the problem attention (Lesson 10) was invented to solve: instead of forcing the decoder to work from one compressed summary, let it look back at *every* encoder hidden state and decide, at each decoding step, which parts of the input are currently most relevant.

See `code/seq2seq_demo.py` for a complete encoder-decoder model trained with teacher forcing on a small synthetic sequence-reversal task, plus a demonstration of the fixed-context bottleneck: performance measured as input sequence length increases.

## Exercises

1. Implement the `Encoder`/`Decoder` pair above and train it with teacher forcing on a task that maps a sequence of digits to its reverse (e.g. "1 2 3" -> "3 2 1"). Confirm it learns the mapping on short sequences.
2. Compare training with teacher forcing against training where the decoder always uses its own previous prediction as input (no teacher forcing). Compare convergence speed.
3. Train the same model on increasingly long input sequences (5, 10, 20 tokens) and measure how accuracy degrades, to directly observe the fixed-context bottleneck.
4. Implement the `<END>` token mechanism and confirm the trained decoder learns to stop generating at approximately the correct length, rather than requiring a hard-coded maximum length cutoff.

## Key Terms

| Term | What it actually means |
|---|---|
| Sequence-to-sequence (seq2seq) | An architecture mapping an input sequence to an independently-lengthed output sequence, via a separate encoder and decoder |
| Encoder | The component of a seq2seq model that reads the entire input sequence and compresses it into a context vector |
| Decoder | The component of a seq2seq model that generates the output sequence one token at a time, conditioned on the encoder's context vector |
| Context vector | The fixed-size vector (the encoder's final hidden state) through which the decoder receives all information about the input sequence |
| Teacher forcing | A training technique that feeds the true previous target token into the decoder, rather than the model's own prediction, stabilizing training |
| Exposure bias | The training/inference mismatch created by teacher forcing, since the decoder never sees its own imperfect predictions during training |
