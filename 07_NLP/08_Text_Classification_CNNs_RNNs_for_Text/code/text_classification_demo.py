"""
TextCNN and TextRNN implemented and trained on the same synthetic
sentence classification dataset, compared on accuracy and training time.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_synthetic_dataset(n=400, seed=0):
    """Two classes distinguished by a local trigram-like pattern
    ('very good'/'very bad') that can appear anywhere in the sentence --
    a good fit for testing whether a model captures LOCAL patterns
    regardless of exact position."""
    rng = np.random.default_rng(seed)
    fillers = ["the", "movie", "was", "quite", "honestly", "really", "totally", "food", "service"]

    texts, labels = [], []
    for _ in range(n):
        label = rng.integers(0, 2)
        marker = "very good" if label == 1 else "very bad"
        n_filler = rng.integers(2, 6)
        words = list(rng.choice(fillers, size=n_filler))
        insert_pos = rng.integers(0, len(words) + 1)
        words[insert_pos:insert_pos] = marker.split()
        texts.append(words)
        labels.append(int(label))

    return texts, labels


def build_vocab(texts):
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in texts:
        for w in t:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def encode_and_pad(texts, vocab, max_len=None):
    if max_len is None:
        max_len = max(len(t) for t in texts)
    encoded = np.zeros((len(texts), max_len), dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, vocab["<unk>"]) for w in t][:max_len]
        encoded[i, :len(ids)] = ids
        lengths[i] = len(ids)
    return torch.from_numpy(encoded), torch.from_numpy(lengths)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, num_filters=8, filter_sizes=(2, 3, 4), num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x).transpose(1, 2)  # (batch, embedding_dim, T)
        conv_outputs = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conv_outputs]
        concatenated = torch.cat(pooled, dim=1)
        return self.fc(concatenated)


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_size=16, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.lstm(packed)
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.fc(h_combined)


def train_and_evaluate(model, X_train, len_train, y_train, X_test, len_test, y_test, n_epochs=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    start = time.perf_counter()
    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train, len_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train, len_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test, len_test).argmax(1) == y_test).float().mean().item()

    return train_acc, test_acc, elapsed


def demo_comparison():
    texts, labels = make_synthetic_dataset(n=400)
    vocab = build_vocab(texts)

    n_train = 300
    X, lengths = encode_and_pad(texts, vocab)
    y = torch.tensor(labels, dtype=torch.long)

    X_train, X_test = X[:n_train], X[n_train:]
    len_train, len_test = lengths[:n_train], lengths[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    torch.manual_seed(0)
    cnn = TextCNN(len(vocab))
    cnn_train_acc, cnn_test_acc, cnn_time = train_and_evaluate(
        cnn, X_train, len_train, y_train, X_test, len_test, y_test
    )

    torch.manual_seed(0)
    rnn = TextRNN(len(vocab))
    rnn_train_acc, rnn_test_acc, rnn_time = train_and_evaluate(
        rnn, X_train, len_train, y_train, X_test, len_test, y_test
    )

    print(f"{'Model':10s} | {'Train acc':10s} | {'Test acc':10s} | {'Train time (s)':15s}")
    print(f"{'TextCNN':10s} | {cnn_train_acc:10.4f} | {cnn_test_acc:10.4f} | {cnn_time:15.4f}")
    print(f"{'TextRNN':10s} | {rnn_train_acc:10.4f} | {rnn_test_acc:10.4f} | {rnn_time:15.4f}")

    print("\nBoth should learn this LOCAL pattern ('very good'/'very bad', appearing")
    print("anywhere in the sentence) well -- exactly the kind of task a CNN's local")
    print("filters are naturally suited to, and note the training TIME difference,")
    print("which reflects the RNN's inherently sequential computation (RNN module")
    print("Lesson 18) versus the CNN's fully parallelizable convolutions.")


if __name__ == "__main__":
    demo_comparison()
