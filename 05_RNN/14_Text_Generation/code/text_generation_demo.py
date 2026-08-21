"""
A character-level language model trained on a small text corpus, with
greedy, temperature-based, and top-k generation compared side by side.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away quickly. "
    "the quick fox and the lazy dog are friends. "
) * 8


def build_char_vocab(text):
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_id, id_to_char


class CharLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        embedded = self.embedding(x)
        out, state = self.rnn(embedded, state)
        logits = self.fc(out)
        return logits, state


def make_training_batches(text, char_to_id, seq_len=40, batch_size=32):
    ids = [char_to_id[c] for c in text]
    inputs, targets = [], []
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        inputs.append(ids[i:i + seq_len])
        targets.append(ids[i + 1:i + seq_len + 1])
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets


def train(model, inputs, targets, n_epochs=60):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 15 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:3d}: loss = {loss.item():.4f}")


def generate_greedy(model, char_to_id, id_to_char, seed, length=80):
    model.eval()
    with torch.no_grad():
        token = torch.tensor([[char_to_id[c] for c in seed]], dtype=torch.long)
        logits, state = model(token)
        next_id = logits[0, -1].argmax().item()
        output = seed + id_to_char[next_id]

        for _ in range(length - len(seed) - 1):
            token = torch.tensor([[next_id]], dtype=torch.long)
            logits, state = model(token, state)
            next_id = logits[0, -1].argmax().item()
            output += id_to_char[next_id]
        return output


def generate_sampled(model, char_to_id, id_to_char, seed, length=80, temperature=1.0, top_k=None):
    model.eval()
    with torch.no_grad():
        token = torch.tensor([[char_to_id[c] for c in seed]], dtype=torch.long)
        logits, state = model(token)
        output = seed

        current_logits = logits[0, -1]
        for _ in range(length - len(seed)):
            scaled = current_logits / temperature
            if top_k is not None:
                top_values, top_indices = scaled.topk(top_k)
                probs_topk = F.softmax(top_values, dim=-1)
                sampled_local_idx = torch.multinomial(probs_topk, 1).item()
                next_id = top_indices[sampled_local_idx].item()
            else:
                probs = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            output += id_to_char[next_id]
            token = torch.tensor([[next_id]], dtype=torch.long)
            logits, state = model(token, state)
            current_logits = logits[0, -1]

        return output


def main():
    torch.manual_seed(0)
    char_to_id, id_to_char = build_char_vocab(CORPUS)
    vocab_size = len(char_to_id)
    print(f"Vocabulary size: {vocab_size} characters\n")

    inputs, targets = make_training_batches(CORPUS, char_to_id)
    model = CharLM(vocab_size)

    print("=== Training character-level language model ===")
    train(model, inputs, targets)

    seed = "the "
    print(f"\n=== Greedy decoding (seed: '{seed}') ===")
    greedy_output = generate_greedy(model, char_to_id, id_to_char, seed)
    print(repr(greedy_output))

    print(f"\n=== Temperature sampling (seed: '{seed}') ===")
    for temp in [0.3, 1.0, 1.5]:
        torch.manual_seed(1)
        out = generate_sampled(model, char_to_id, id_to_char, seed, temperature=temp)
        print(f"T={temp}: {out!r}")

    print(f"\n=== Top-k sampling (seed: '{seed}', temperature=1.0) ===")
    for k in [2, 5, vocab_size]:
        torch.manual_seed(1)
        out = generate_sampled(model, char_to_id, id_to_char, seed, temperature=1.0, top_k=k)
        label = f"k={k}" if k < vocab_size else "k=vocab_size (no restriction)"
        print(f"{label}: {out!r}")


if __name__ == "__main__":
    main()
