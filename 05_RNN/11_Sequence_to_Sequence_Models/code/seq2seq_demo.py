"""
A complete seq2seq model trained on a small synthetic "reverse the
sequence" task, with teacher forcing during training and free-running
(no teacher forcing) generation at inference time.
"""

import numpy as np
import torch
import torch.nn as nn

PAD, START, END = 0, 1, 2
VOCAB_OFFSET = 3  # digit tokens start at index 3
VOCAB_SIZE = VOCAB_OFFSET + 10  # digits 0-9


def make_reversal_example(min_len=3, max_len=6, rng=None):
    rng = rng or np.random.default_rng()
    length = rng.integers(min_len, max_len + 1)
    digits = rng.integers(0, 10, size=length)
    source = [d + VOCAB_OFFSET for d in digits]
    target = list(reversed(source))
    return source, target


def make_batch(n, min_len=3, max_len=6, seed=0):
    rng = np.random.default_rng(seed)
    sources, targets = [], []
    for _ in range(n):
        s, t = make_reversal_example(min_len, max_len, rng)
        sources.append(s)
        targets.append(t)
    return sources, targets


def pad_batch(sequences, add_start=False, add_end=False):
    seqs = []
    for s in sequences:
        seq = s.copy()
        if add_start:
            seq = [START] + seq
        if add_end:
            seq = seq + [END]
        seqs.append(seq)
    max_len = max(len(s) for s in seqs)
    padded = [s + [PAD] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, source_seq, decoder_input_seq):
        source_embedded = self.embedding(source_seq)
        _, (h, c) = self.encoder(source_embedded)

        decoder_embedded = self.embedding(decoder_input_seq)
        decoder_output, _ = self.decoder(decoder_embedded, (h, c))
        return self.output_layer(decoder_output)

    def generate(self, source_seq, max_len=10):
        """Free-running generation: no teacher forcing, feed own predictions back in."""
        self.eval()
        with torch.no_grad():
            source_embedded = self.embedding(source_seq)
            _, (h, c) = self.encoder(source_embedded)

            token = torch.tensor([[START]], dtype=torch.long)
            outputs = []
            for _ in range(max_len):
                embedded = self.embedding(token)
                out, (h, c) = self.decoder(embedded, (h, c))
                logits = self.output_layer(out)
                next_token = logits.argmax(dim=-1)
                token_id = next_token.item()
                if token_id == END:
                    break
                outputs.append(token_id)
                token = next_token
            return outputs


def train(model, n_steps=400, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for step in range(n_steps):
        sources, targets = make_batch(batch_size, seed=step)
        source_tensor = pad_batch(sources)
        # Teacher forcing: decoder INPUT is <START> + target[:-1]; decoder TARGET is target + <END>
        decoder_input = pad_batch(targets, add_start=True)[:, :-1]
        decoder_target = pad_batch(targets, add_end=True)[:, :decoder_input.shape[1]]

        optimizer.zero_grad()
        logits = model(source_tensor, decoder_input)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), decoder_target.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            print(f"step {step:4d}: loss = {loss.item():.4f}")


def evaluate_reversal_accuracy(model, n_examples=50, min_len=3, max_len=6):
    rng = np.random.default_rng(999)
    correct = 0
    for _ in range(n_examples):
        source, target = make_reversal_example(min_len, max_len, rng)
        source_tensor = torch.tensor([source], dtype=torch.long)
        generated = model.generate(source_tensor, max_len=max_len + 2)
        if generated == target:
            correct += 1
    return correct / n_examples


def main():
    torch.manual_seed(0)
    model = Seq2Seq(vocab_size=VOCAB_SIZE, embedding_dim=16, hidden_size=32)

    print("=== Training on 'reverse the sequence' task (teacher forcing) ===")
    train(model, n_steps=1200)

    print("\n=== Free-running generation (no teacher forcing) on new examples ===")
    rng = np.random.default_rng(42)
    for _ in range(5):
        source, target = make_reversal_example(rng=rng)
        source_tensor = torch.tensor([source], dtype=torch.long)
        generated = model.generate(source_tensor, max_len=8)
        source_digits = [t - VOCAB_OFFSET for t in source]
        target_digits = [t - VOCAB_OFFSET for t in target]
        generated_digits = [t - VOCAB_OFFSET for t in generated]
        match = "OK" if generated_digits == target_digits else "MISS"
        print(f"input={source_digits}  target={target_digits}  generated={generated_digits}  [{match}]")

    print("\n=== Overall accuracy on 50 held-out examples ===")
    acc = evaluate_reversal_accuracy(model)
    print(f"Exact-match accuracy: {acc:.2%}")
    print("(This is a small model trained briefly for a fast, illustrative demo --")
    print(" not a production training run. The mechanism -- teacher forcing during")
    print(" training, free-running generation at inference -- is what matters here;")
    print(" more steps/capacity would push accuracy substantially higher.)")


if __name__ == "__main__":
    main()
