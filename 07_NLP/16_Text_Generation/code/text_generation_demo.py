"""
A character-level LSTM language model trained on a small synthetic
corpus, with greedy, temperature, and top-k decoding compared side by
side, including a demonstration of greedy decoding's repetition-loop
failure mode.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


def build_char_vocab(text):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def make_training_batches(text, char_to_idx, seq_len=40, batch_size=32):
    ids = [char_to_idx[c] for c in text]
    inputs, targets = [], []
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        inputs.append(ids[i:i + seq_len])
        targets.append(ids[i + 1:i + seq_len + 1])
    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    return X, Y


def train_char_lm(text, n_epochs=150, lr=0.005, seed=0):
    torch.manual_seed(seed)
    char_to_idx, idx_to_char = build_char_vocab(text)
    X, Y = make_training_batches(text, char_to_idx)

    model = CharLM(len(char_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = loss_fn(logits.reshape(-1, len(char_to_idx)), Y.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  loss: {loss.item():.4f}")

    return model, char_to_idx, idx_to_char


def generate_greedy(model, seed_text, char_to_idx, idx_to_char, length=100):
    model.eval()
    ids = [char_to_idx[c] for c in seed_text]
    hidden = None
    generated = list(seed_text)
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        logits, hidden = model(x, hidden)
        for _ in range(length):
            next_id = logits[0, -1].argmax().item()
            generated.append(idx_to_char[next_id])
            x = torch.tensor([[next_id]], dtype=torch.long)
            logits, hidden = model(x, hidden)
    return "".join(generated)


def generate_sampled(model, seed_text, char_to_idx, idx_to_char, length=100,
                      temperature=1.0, top_k=None, seed=0):
    torch.manual_seed(seed)
    model.eval()
    ids = [char_to_idx[c] for c in seed_text]
    hidden = None
    generated = list(seed_text)
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        logits, hidden = model(x, hidden)
        for _ in range(length):
            step_logits = logits[0, -1] / temperature
            if top_k is not None:
                top_vals, top_idx = step_logits.topk(top_k)
                probs = F.softmax(top_vals, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                next_id = top_idx[sampled].item()
            else:
                probs = F.softmax(step_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(idx_to_char[next_id])
            x = torch.tensor([[next_id]], dtype=torch.long)
            logits, hidden = model(x, hidden)
    return "".join(generated)


def demo_text_generation():
    # A small, repetitive synthetic corpus (short sentences repeated with variation)
    corpus = (
        "the cat sat on the mat. the dog sat on the rug. "
        "the cat ran to the park. the dog ran to the yard. "
        "the cat likes the mat. the dog likes the rug. "
    ) * 20

    print("=== Training character-level language model ===")
    model, char_to_idx, idx_to_char = train_char_lm(corpus, n_epochs=150)

    seed_text = "the cat"
    print(f"\n=== Greedy decoding (seed: '{seed_text}') ===")
    greedy_output = generate_greedy(model, seed_text, char_to_idx, idx_to_char, length=80)
    print(repr(greedy_output))
    print("(Watch for a repetition loop -- greedy decoding has no mechanism to escape")
    print("once it settles into repeating a high-confidence phrase pattern.)\n")

    print("=== Temperature sampling comparison ===")
    for temp in [0.3, 0.8, 1.5]:
        output = generate_sampled(model, seed_text, char_to_idx, idx_to_char,
                                   length=80, temperature=temp, seed=1)
        print(f"temperature={temp}: {repr(output)}")

    print("\n=== Top-k sampling comparison ===")
    for k in [3, 10, len(char_to_idx)]:
        output = generate_sampled(model, seed_text, char_to_idx, idx_to_char,
                                   length=80, temperature=0.8, top_k=k, seed=1)
        print(f"top_k={k}: {repr(output)}")

    print("\nLower temperature / smaller top_k stays closer to the training corpus's")
    print("safe, repeated patterns; higher temperature / larger top_k (effectively no")
    print("restriction) introduces more variety, at increasing risk of incoherence.")


if __name__ == "__main__":
    demo_text_generation()
