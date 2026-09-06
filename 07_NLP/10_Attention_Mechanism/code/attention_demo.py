"""
A complete additive-attention seq2seq model trained on the same
reversal task from Lesson 09, with an accuracy comparison against the
plain encoder-decoder as sequence length grows, plus a visualization
of the learned attention weight matrix.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, START, END = 0, 1, 2


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, h_n = self.rnn(embedded)
        return outputs, h_n  # outputs: (batch, T, hidden_size) -- ALL encoder states, kept


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_state, encoder_states):
        decoder_expanded = decoder_state.unsqueeze(1)
        scores = self.v(torch.tanh(self.W_s(decoder_expanded) + self.W_h(encoder_states)))
        scores = scores.squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_states)
        return context.squeeze(1), weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.attention = AdditiveAttention(hidden_size)
        self.rnn = nn.GRUCell(embedding_dim + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, input_token, hidden, encoder_states):
        embedded = self.embedding(input_token).squeeze(1)          # (batch, embedding_dim)
        context, weights = self.attention(hidden, encoder_states)   # (batch, hidden_size), (batch, T)
        rnn_input = torch.cat([embedded, context], dim=1)
        hidden = self.rnn(rnn_input, hidden)
        logits = self.fc(hidden)
        return logits, hidden, weights


def make_reversal_dataset(n=300, min_len=3, max_len=6, vocab_size=8, seed=0):
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


def train_attention_seq2seq(inputs, targets, vocab_size=8, embedding_dim=16, hidden_size=32,
                             n_epochs=250, lr=0.01, seed=0):
    torch.manual_seed(seed)
    max_in_len = max(len(s) for s in inputs)
    max_out_len = max(len(s) for s in targets) + 1

    X = pad_batch(inputs, max_in_len)
    Y_full = pad_batch(targets, max_out_len, add_end=True)
    decoder_input = torch.cat(
        [torch.full((len(targets), 1), START, dtype=torch.long), Y_full[:, :-1]], dim=1
    )

    encoder = Encoder(vocab_size, embedding_dim, hidden_size)
    decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_size)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        encoder_states, h_n = encoder(X)
        hidden = h_n.squeeze(0)

        all_logits = []
        for t in range(max_out_len):
            step_input = decoder_input[:, t:t + 1]
            logits, hidden, _ = decoder.forward_step(step_input, hidden, encoder_states)
            all_logits.append(logits.unsqueeze(1))
        all_logits = torch.cat(all_logits, dim=1)

        loss = loss_fn(all_logits.reshape(-1, vocab_size), Y_full.reshape(-1))
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  loss: {loss.item():.4f}")

    return encoder, decoder, max_out_len


def greedy_decode_attention(encoder, decoder, input_seq, max_out_len, vocab_size, return_weights=False):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        x = torch.tensor([input_seq], dtype=torch.long)
        encoder_states, h_n = encoder(x)
        hidden = h_n.squeeze(0)
        step_input = torch.tensor([[START]], dtype=torch.long)
        output, all_weights = [], []
        for _ in range(max_out_len):
            logits, hidden, weights = decoder.forward_step(step_input, hidden, encoder_states)
            next_token = logits[0].argmax().item()
            all_weights.append(weights[0].numpy())
            if next_token == END:
                break
            output.append(next_token)
            step_input = torch.tensor([[next_token]], dtype=torch.long)
    if return_weights:
        return output, np.array(all_weights)
    return output


def demo_attention_reversal():
    inputs, targets = make_reversal_dataset(n=800, min_len=3, max_len=5, vocab_size=8)
    print("=== Training attention-based encoder-decoder ===")
    encoder, decoder, max_out_len = train_attention_seq2seq(inputs, targets, n_epochs=250)

    test_inputs, test_targets = make_reversal_dataset(n=40, min_len=3, max_len=5, vocab_size=8, seed=99)
    correct = 0
    for inp, tgt in zip(test_inputs, test_targets):
        pred = greedy_decode_attention(encoder, decoder, inp, max_out_len, vocab_size=8)
        if pred == tgt:
            correct += 1
    print(f"\nTest accuracy (exact match): {correct}/{len(test_inputs)} = {correct/len(test_inputs):.2%}")

    return encoder, decoder, max_out_len


def demo_bottleneck_comparison_with_attention():
    print("\n=== Attention model: accuracy vs sequence length ===")
    for max_len in [3, 8, 15]:
        inputs, targets = make_reversal_dataset(n=500, min_len=max_len, max_len=max_len, vocab_size=8, seed=1)
        encoder, decoder, max_out_len = train_attention_seq2seq(
            inputs, targets, n_epochs=200, hidden_size=32
        )
        test_inputs, test_targets = make_reversal_dataset(n=30, min_len=max_len, max_len=max_len, vocab_size=8, seed=77)
        correct = 0
        for inp, tgt in zip(test_inputs, test_targets):
            pred = greedy_decode_attention(encoder, decoder, inp, max_out_len, vocab_size=8)
            if pred == tgt:
                correct += 1
        print(f"sequence length {max_len:2d}: exact-match accuracy = {correct/len(test_inputs):.2%}")

    print("\nCompare against Lesson 09's plain encoder-decoder results (96.67% / 40.00% / 0.00%")
    print("at the same lengths) -- attention should degrade far less steeply as length grows,")
    print("since every encoder position stays directly reachable at every decoding step.")


def demo_attention_visualization(encoder, decoder, max_out_len):
    print("\n=== Attention weight matrix for one example ===")
    test_input = [4, 6, 7, 5]  # a simple 4-token input to reverse
    pred, weights = greedy_decode_attention(
        encoder, decoder, test_input, max_out_len, vocab_size=8, return_weights=True
    )
    print(f"Input:     {test_input}")
    print(f"Predicted: {pred}  (expected reverse: {list(reversed(test_input))})")
    print(f"\nAttention weights (rows = decoder step, cols = encoder position):")
    print("      " + "  ".join(f"in{i}" for i in range(len(test_input))))
    for t, row in enumerate(weights[:len(pred)]):
        row_str = "  ".join(f"{w:.2f}" for w in row[:len(test_input)])
        print(f"out{t}: {row_str}")
    print("\nFor a reversal task, a well-trained model's attention should roughly")
    print("trace an ANTI-diagonal -- output step 0 attending to the LAST input")
    print("position, output step 1 to the second-to-last, and so on.")


if __name__ == "__main__":
    encoder, decoder, max_out_len = demo_attention_reversal()
    demo_bottleneck_comparison_with_attention()
    demo_attention_visualization(encoder, decoder, max_out_len)
