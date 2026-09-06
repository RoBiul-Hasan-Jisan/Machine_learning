"""
A basic encoder-decoder seq2seq model trained with teacher forcing on a
synthetic sequence-reversal task, plus a demonstration of the fixed-
context-vector bottleneck as input length grows.
"""

import numpy as np
import torch
import torch.nn as nn

PAD, START, END = 0, 1, 2


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, h_n = self.rnn(embedded)
        return h_n


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


def make_reversal_dataset(n=300, min_len=3, max_len=6, vocab_size=10, seed=0):
    """Map a sequence of digit-tokens to its reverse. Tokens 3..vocab_size-1
    are digits (0,1,2 reserved for PAD/START/END)."""
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for _ in range(n):
        length = rng.integers(min_len, max_len + 1)
        seq = list(rng.integers(3, vocab_size, size=length))
        inputs.append(seq)
        targets.append(list(reversed(seq)))
    return inputs, targets


def pad_batch(sequences, max_len, add_end=False):
    padded = np.full((len(sequences), max_len), PAD, dtype=np.int64)
    for i, seq in enumerate(sequences):
        s = seq + [END] if add_end else seq
        s = s[:max_len]
        padded[i, :len(s)] = s
    return torch.from_numpy(padded)


def train_seq2seq(inputs, targets, vocab_size=10, embedding_dim=16, hidden_size=32,
                   n_epochs=200, lr=0.01, teacher_forcing=True, seed=0):
    torch.manual_seed(seed)
    max_in_len = max(len(s) for s in inputs)
    max_out_len = max(len(s) for s in targets) + 1  # +1 for END token

    X = pad_batch(inputs, max_in_len)
    Y_full = pad_batch(targets, max_out_len, add_end=True)  # decoder targets (include END)
    decoder_input = torch.cat(
        [torch.full((len(targets), 1), START, dtype=torch.long), Y_full[:, :-1]], dim=1
    )

    encoder = Encoder(vocab_size, embedding_dim, hidden_size)
    decoder = Decoder(vocab_size, embedding_dim, hidden_size)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        context = encoder(X)

        if teacher_forcing:
            logits, _ = decoder(decoder_input, context)
        else:
            # feed the decoder's OWN previous prediction instead of the true target
            hidden = context
            step_input = torch.full((len(targets), 1), START, dtype=torch.long)
            all_logits = []
            for t in range(max_out_len):
                step_logits, hidden = decoder(step_input, hidden)
                all_logits.append(step_logits)
                step_input = step_logits.argmax(dim=-1).detach()
            logits = torch.cat(all_logits, dim=1)

        loss = loss_fn(logits.reshape(-1, vocab_size), Y_full.reshape(-1))
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  loss: {loss.item():.4f}")

    return encoder, decoder, max_out_len


def greedy_decode(encoder, decoder, input_seq, max_out_len, vocab_size):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        x = torch.tensor([input_seq], dtype=torch.long)
        hidden = encoder(x)
        step_input = torch.tensor([[START]], dtype=torch.long)
        output = []
        for _ in range(max_out_len):
            logits, hidden = decoder(step_input, hidden)
            next_token = logits[0, 0].argmax().item()
            if next_token == END:
                break
            output.append(next_token)
            step_input = torch.tensor([[next_token]], dtype=torch.long)
    return output


def demo_seq2seq_reversal():
    inputs, targets = make_reversal_dataset(n=800, min_len=3, max_len=5, vocab_size=8)
    print("=== Training encoder-decoder with teacher forcing ===")
    encoder, decoder, max_out_len = train_seq2seq(
        inputs, targets, vocab_size=8, hidden_size=48, n_epochs=400
    )

    print("\n=== Example predictions ===")
    test_inputs, test_targets = make_reversal_dataset(n=40, min_len=3, max_len=5, vocab_size=8, seed=99)
    for inp, tgt in zip(test_inputs[:8], test_targets[:8]):
        pred = greedy_decode(encoder, decoder, inp, max_out_len, vocab_size=8)
        match = "OK" if pred == tgt else "WRONG"
        print(f"input={list(map(int, inp))}  target={list(map(int, tgt))}  predicted={pred}  [{match}]")

    correct = 0
    for inp, tgt in zip(test_inputs, test_targets):
        pred = greedy_decode(encoder, decoder, inp, max_out_len, vocab_size=8)
        if pred == tgt:
            correct += 1
    print(f"\nFull test accuracy (exact sequence match): {correct}/{len(test_inputs)} = {correct/len(test_inputs):.2%}")


def demo_context_bottleneck():
    """Train separately on short vs long sequences and compare exact-match
    accuracy, illustrating the fixed-context-vector bottleneck."""
    print("\n=== Fixed-context bottleneck: accuracy vs sequence length ===")
    for max_len in [3, 8, 15]:
        inputs, targets = make_reversal_dataset(n=500, min_len=max_len, max_len=max_len, vocab_size=8, seed=1)
        encoder, decoder, max_out_len = train_seq2seq(
            inputs, targets, vocab_size=8, n_epochs=250, hidden_size=32
        )

        test_inputs, test_targets = make_reversal_dataset(n=30, min_len=max_len, max_len=max_len, vocab_size=8, seed=77)
        correct = 0
        for inp, tgt in zip(test_inputs, test_targets):
            pred = greedy_decode(encoder, decoder, inp, max_out_len, vocab_size=8)
            if pred == tgt:
                correct += 1
        print(f"sequence length {max_len:2d}: exact-match accuracy = {correct/len(test_inputs):.2%}")

    print("\nAs sequence length grows, the SAME fixed-size context vector must")
    print("summarize more information, and accuracy typically degrades --")
    print("exactly the bottleneck attention (Lesson 10) is designed to remove.")


if __name__ == "__main__":
    demo_seq2seq_reversal()
    demo_context_bottleneck()
