"""
From-scratch attention (additive/Bahdanau-style), integrated into an
attention-augmented decoder trained on the reversal task from Lesson 11,
with a printed visualization of the resulting attention weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, START, END = 0, 1, 2
VOCAB_OFFSET = 3
VOCAB_SIZE = VOCAB_OFFSET + 10


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def attention(decoder_state, encoder_states, W_s, W_h, v):
    scores = np.array([
        v @ np.tanh(W_s @ decoder_state + W_h @ h_i) for h_i in encoder_states
    ])
    weights = softmax(scores)
    context = sum(w * h for w, h in zip(weights, encoder_states))
    return context, weights


def demo_attention_from_scratch():
    rng = np.random.default_rng(0)
    hidden_size = 4
    T = 5

    W_s = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.3
    W_h = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.3
    v = rng.normal(size=hidden_size).astype(np.float32) * 0.3

    encoder_states = [rng.normal(size=hidden_size).astype(np.float32) for _ in range(T)]
    decoder_state = rng.normal(size=hidden_size).astype(np.float32)

    context, weights = attention(decoder_state, encoder_states, W_s, W_h, v)
    print("Attention weights:", weights.round(4))
    print("Sum of weights (should be 1.0):", round(weights.sum(), 6))
    print("Context vector shape:", context.shape, "\n")
    assert abs(weights.sum() - 1.0) < 1e-5


class AttentionDecoder(nn.Module):
    """A decoder that, at each step, attends over ALL encoder hidden states
    rather than relying on a single fixed context vector (Lesson 11)."""

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # Additive attention parameters
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        self.decoder_cell = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def compute_attention(self, decoder_state, encoder_states):
        # decoder_state: (batch, H). encoder_states: (batch, T, H)
        query = self.W_s(decoder_state).unsqueeze(1)          # (batch, 1, H)
        keys = self.W_h(encoder_states)                        # (batch, T, H)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)  # (batch, T)
        weights = F.softmax(scores, dim=-1)                    # (batch, T)
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)  # (batch, H)
        return context, weights

    def forward(self, source_seq, decoder_input_seq):
        source_embedded = self.embedding(source_seq)
        encoder_states, (h, c) = self.encoder(source_embedded)  # encoder_states: (batch, T, H)
        h, c = h.squeeze(0), c.squeeze(0)

        decoder_embedded = self.embedding(decoder_input_seq)  # (batch, T_dec, D)
        outputs = []
        attention_weights_all = []
        for t in range(decoder_embedded.shape[1]):
            context, weights = self.compute_attention(h, encoder_states)
            cell_input = torch.cat([decoder_embedded[:, t, :], context], dim=-1)
            h, c = self.decoder_cell(cell_input, (h, c))
            outputs.append(self.output_layer(h))
            attention_weights_all.append(weights)

        return torch.stack(outputs, dim=1), torch.stack(attention_weights_all, dim=1)


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


def train_attention_decoder(model, n_steps=500, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for step in range(n_steps):
        sources, targets = make_batch(batch_size, min_len=4, max_len=4, seed=step)  # fixed length for a clean attention demo
        source_tensor = pad_batch(sources)
        decoder_input = pad_batch(targets, add_start=True)[:, :-1]
        decoder_target = pad_batch(targets, add_end=True)[:, :decoder_input.shape[1]]

        optimizer.zero_grad()
        logits, _ = model(source_tensor, decoder_input)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), decoder_target.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            print(f"step {step:4d}: loss = {loss.item():.4f}")


def demo_visualize_attention(model):
    rng = np.random.default_rng(7)
    source, target = make_reversal_example(min_len=4, max_len=4, rng=rng)
    source_tensor = pad_batch([source])
    decoder_input = pad_batch([target], add_start=True)[:, :-1]

    model.eval()
    with torch.no_grad():
        _, attn_weights = model(source_tensor, decoder_input)
    attn_weights = attn_weights.squeeze(0).numpy()  # (T_dec, T_enc)

    source_digits = [t - VOCAB_OFFSET for t in source]
    target_digits = [t - VOCAB_OFFSET for t in target]
    print(f"\nInput sequence:  {source_digits}")
    print(f"Target (reversed): {target_digits}\n")
    print("Attention weights (rows = decoder step, cols = encoder position):")
    header = "        " + "".join(f"pos{i}  " for i in range(len(source)))
    print(header)
    for t, row in enumerate(attn_weights):
        row_str = "  ".join(f"{w:.2f}" for w in row)
        print(f"step {t}:  {row_str}   (should peak near encoder position {len(source) - 1 - t})")


if __name__ == "__main__":
    print("=== Attention from scratch ===")
    demo_attention_from_scratch()

    print("=== Training an attention-augmented decoder on the reversal task ===")
    torch.manual_seed(0)
    model = AttentionDecoder(vocab_size=VOCAB_SIZE, embedding_dim=16, hidden_size=24)
    train_attention_decoder(model, n_steps=500)

    print("\n=== Visualizing learned attention weights ===")
    demo_visualize_attention(model)
