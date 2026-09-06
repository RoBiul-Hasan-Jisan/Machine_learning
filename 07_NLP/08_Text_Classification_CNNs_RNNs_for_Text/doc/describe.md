# 08. Text Classification — CNNs & RNNs for Text

## Learning Objectives

- Adapt 1D convolution to text and explain what a text CNN's filters detect
- Build an RNN/LSTM text classifier and compare it against a CNN-based one
- Know the practical tradeoffs that determine which architecture suits a given text classification task

## The Problem

Lesson 05's TF-IDF + logistic regression baseline is fast and often strong, but it has a hard ceiling: it can't use word order (Lesson 02's core limitation) or exploit pretrained embeddings' semantic structure (Lessons 03-04) beyond whatever a single linear layer can extract from a static count vector. Deep architectures — CNNs and RNNs, both covered in depth in their own modules — adapt naturally to text once you settle on the right way to represent a sequence of words as a network's input, unlocking word-order sensitivity and contextual representations a bag-of-words approach cannot provide.

## The Concept

### From images to text: 1D convolution over word embeddings

The CNN module's convolution operation slides a 2D filter over a 2D grid of pixels. For text, replace the image with a sequence of word embeddings — a `T × embedding_dim` matrix (T words, each an `embedding_dim`-length vector, from Lesson 03/04) — and slide a **1D** filter down the sequence dimension, always spanning the *entire* embedding dimension at each position:

```
Embedded sentence (T=6 words, embedding_dim=4):

[ e_the  ]
[ e_dog  ]
[ e_ran  ]
[ e_very ]
[ e_fast ]
[ e_.    ]

A filter of "width" 3 (spanning 3 consecutive words, full embedding_dim) slides down:
  position 0: covers [e_the, e_dog, e_ran]     -> one output value
  position 1: covers [e_dog, e_ran, e_very]    -> one output value
  position 2: covers [e_ran, e_very, e_fast]   -> one output value
  ...
```

Each filter acts as a detector for a specific local word-pattern — a filter width of 3 detects trigram-like patterns (analogous to Lesson 02's n-grams, but as a *learned* pattern rather than an explicit discrete feature), and using several filters of different widths (e.g. widths 2, 3, 4 in parallel, each with many filters) lets the network detect patterns at several local scales simultaneously — directly echoing the CNN module's Lesson 03 (filters as pattern detectors) and its GoogLeNet lesson's multi-scale idea, applied to text instead of images.

### Max-over-time pooling: one output value per filter

After convolving, each filter produces one score at every position in the sequence — but sentences have varying length, and the classifier needs a single fixed-size vector regardless of length. **Max-over-time pooling** solves this exactly the way the CNN module's max pooling did (Lesson 05 there): take the single largest activation across the *entire* sequence for each filter, discarding position information entirely but keeping "did this filter's pattern fire strongly anywhere in this sentence":

```python
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)      # (batch, embedding_dim, T)
        conv_outputs = [F.relu(conv(embedded)) for conv in self.convs]   # each: (batch, num_filters, T')
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conv_outputs]  # max-over-time
        concatenated = torch.cat(pooled, dim=1)             # (batch, num_filters * len(filter_sizes))
        return self.fc(concatenated)
```

This is the "TextCNN" architecture (Kim, 2014), a fast, strong baseline for sentence classification that captures local n-gram-like patterns without needing sequential (step-by-step) computation at all — every filter position can be computed in parallel, similar in spirit to the parallelization advantage the RNN module's Lesson 18 highlighted for attention-based models.

### RNN/LSTM text classification

The RNN module's Lesson 04 (many-to-one pattern) and Lesson 09 (bidirectionality) apply directly to text classification: embed each token, run the sequence through an LSTM or GRU, and classify from the final (or bidirectionally-concatenated) hidden state:

```python
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.fc(h_combined)
```

This is precisely the RNN module's Lesson 19 Project 1 (sentiment/document classification), described there in full — this lesson exists to place it side-by-side with the CNN alternative and make the comparison explicit.

### CNN vs RNN for text: practical tradeoffs

| | Text CNN | Text RNN (LSTM/GRU) |
|---|---|---|
| Captures | Local n-gram-like patterns, at whatever filter widths are chosen | Sequential dependencies across the whole sequence, in principle unbounded range |
| Parallelizable during training | Yes — every filter position computed independently | No — strictly sequential (the RNN module's Lesson 18 point) |
| Long-range dependencies | Limited to filter width unless stacked/dilated | Better suited in principle, though vanishing gradients (RNN module Lesson 06) still apply |
| Typical speed | Faster to train | Slower, due to sequential computation |
| Common strength | Sentence/short-document classification (sentiment, topic) where local patterns dominate | Tasks where longer-range structure or precise sequence modeling matters |

In practice, both are strong, fast baselines that are increasingly superseded by pretrained Transformer-based encoders (the embedding models covered in Lesson 22) for tasks with enough data and compute budget to fine-tune one — but a TextCNN or TextRNN remains a very reasonable, cheap, well-understood choice when a full Transformer is unnecessary overhead, particularly for shorter texts where a Transformer's extra capacity for long-range context provides little benefit.

See `code/text_classification_demo.py` for both a TextCNN and a TextRNN implemented and trained on the same synthetic classification dataset, compared directly on accuracy and training time.

## Exercises

1. Implement `TextCNN` with filter widths [2, 3, 4] and train it on a small synthetic sentence classification dataset. Inspect which n-gram-like patterns (by finding which input positions maximize a given filter's activation) a few filters seem to have learned.
2. Compare `TextCNN` and `TextRNN` on the same dataset and the same number of training epochs — report accuracy and wall-clock training time for each.
3. Construct a classification task where word order over a long distance matters (e.g. "not only X but also Y" vs "Y but not only X" with different labels) and compare how well each architecture handles it.
4. Stack two TextCNN filter layers (a second `Conv1d` applied to the first layer's output) and explain, using the CNN module's Lesson 07 (CNN Architecture) reasoning, how this increases the effective range of context each unit can capture.

## Key Terms

| Term | What it actually means |
|---|---|
| TextCNN | A convolutional architecture for text that slides 1D filters across a sequence of word embeddings to detect local n-gram-like patterns |
| Max-over-time pooling | Taking the maximum filter activation across the entire sequence length, producing a fixed-size vector regardless of sentence length |
| Filter width | In a text CNN, the number of consecutive words (tokens) a single filter spans at each position |
