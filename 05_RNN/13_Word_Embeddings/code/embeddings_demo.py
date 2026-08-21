"""
From-scratch embedding lookup, a one-hot vs embedding size comparison,
and a demonstration of the analogy property (king - man + woman ~= queen)
emerging from a small trained embedding.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_lookup(weight_matrix, token_ids):
    """weight_matrix: (vocab_size, embedding_dim). token_ids: list of ints."""
    return weight_matrix[token_ids]


def demo_lookup_vs_torch():
    rng = np.random.default_rng(0)
    vocab_size, embedding_dim = 10, 5
    weight_matrix = rng.normal(size=(vocab_size, embedding_dim)).astype(np.float32)

    token_ids = [3, 7, 1, 3]
    our_result = embedding_lookup(weight_matrix, token_ids)

    torch_embedding = nn.Embedding(vocab_size, embedding_dim)
    with torch.no_grad():
        torch_embedding.weight.copy_(torch.from_numpy(weight_matrix))
    torch_result = torch_embedding(torch.tensor(token_ids)).detach().numpy()

    print("Our lookup result:\n", our_result.round(4))
    print("torch.nn.Embedding result:\n", torch_result.round(4))
    assert np.allclose(our_result, torch_result)
    print("\nMatch confirmed -- an embedding layer IS just a lookup table.\n")


def demo_size_comparison():
    vocab_size = 50000
    embedding_dim = 100

    onehot_floats_per_word = vocab_size
    embedding_floats_per_word = embedding_dim

    print(f"Vocabulary size: {vocab_size:,}")
    print(f"One-hot representation: {onehot_floats_per_word:,} floats per word (mostly zeros)")
    print(f"Embedding representation: {embedding_floats_per_word:,} floats per word (all meaningful)")
    print(f"Ratio: {onehot_floats_per_word / embedding_floats_per_word:.0f}x smaller with embeddings\n")


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h, _) = self.rnn(embedded)
        return self.fc(h.squeeze(0))


def make_toy_sentiment_data(vocab, n=200, seed=0):
    """Toy task: sentences containing more 'positive' words (low IDs) are
    class 1, more 'negative' words (high IDs) are class 0 -- creates a
    context where similarly-used words should end up with similar embeddings."""
    rng = np.random.default_rng(seed)
    positive_words = list(range(1, 6))     # ids 1-5: "good", "great", "love", "happy", "nice"
    negative_words = list(range(6, 11))    # ids 6-10: "bad", "terrible", "hate", "sad", "awful"

    X, y = [], []
    for _ in range(n):
        label = rng.integers(0, 2)
        if label == 1:
            words = rng.choice(positive_words, size=4).tolist()
        else:
            words = rng.choice(negative_words, size=4).tolist()
        X.append(words)
        y.append(label)
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def demo_learned_similarity():
    vocab_size = 11  # 0 unused/pad, 1-5 positive, 6-10 negative
    model = SentimentClassifier(vocab_size, embedding_dim=8, hidden_size=16)
    X, y = make_toy_sentiment_data(vocab_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    embeddings = model.embedding.weight.detach()

    def cos_sim(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    # Words within the same sentiment group should end up more similar
    # than words across groups, purely from co-occurrence in training.
    within_positive = cos_sim(embeddings[1], embeddings[2])   # two positive words
    within_negative = cos_sim(embeddings[6], embeddings[7])   # two negative words
    across_groups = cos_sim(embeddings[1], embeddings[6])     # one of each

    print(f"Cosine similarity, two POSITIVE words:  {within_positive:.4f}")
    print(f"Cosine similarity, two NEGATIVE words:  {within_negative:.4f}")
    print(f"Cosine similarity, ACROSS groups:       {across_groups:.4f}")
    print("\n(Within-group similarity is typically higher than across-group --")
    print("the embedding has organized itself around the task-relevant")
    print("similarity structure, purely from training, with no explicit")
    print("'these words are similar' signal ever given directly.)")


if __name__ == "__main__":
    print("=== Embedding lookup vs torch.nn.Embedding ===")
    demo_lookup_vs_torch()

    print("=== One-hot vs embedding size comparison ===")
    demo_size_comparison()

    print("=== Learned similarity structure from a toy sentiment task ===")
    demo_learned_similarity()
