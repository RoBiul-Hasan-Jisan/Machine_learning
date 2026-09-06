"""
A from-scratch BLEU score implementation, a beam search decoder built
on top of Lesson 10's attention-based seq2seq model, and a comparison
of greedy vs beam search decoding on the reversal task.
"""

import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, START, END = 0, 1, 2


# --- BLEU score from scratch ---

def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(candidate, reference, max_n=4):
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    if len(candidate_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate_tokens, n)
        ref_ngrams = get_ngrams(reference_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(overlap / total if total > 0 else 0.0)

    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    brevity_penalty = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(1 - len(reference_tokens) / len(candidate_tokens))

    return brevity_penalty * geo_mean


def demo_bleu():
    reference = "the cat sat on the mat"
    candidates = [
        "the cat sat on the mat",             # exact match
        "the cat was sitting on the mat",     # valid paraphrase, different words
        "a dog ran in the park",              # unrelated
        "the cat",                             # too short, heavily penalized
    ]

    print("Reference:", reference)
    print("(using BLEU-2, i.e. unigram + bigram precision, for a less all-or-nothing score)")
    for c in candidates:
        score = bleu_score(c, reference, max_n=2)
        print(f"  candidate: '{c}'  ->  BLEU-2 = {score:.4f}")

    print("\nNote: the exact match scores 1.0, but the VALID paraphrase ('was sitting'")
    print("instead of 'sat') scores meaningfully lower despite being an equally correct")
    print("translation -- BLEU only rewards surface n-gram overlap with the specific")
    print("reference wording, not semantic correctness. (With the standard BLEU-4,")
    print("short sentences like these often collapse straight to 0 the moment ANY")
    print("4-gram fails to match -- itself a well-known limitation for short text.)\n")


# --- Reused attention seq2seq model (from Lesson 10) ---

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, h_n = self.rnn(embedded)
        return outputs, h_n


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
        embedded = self.embedding(input_token).squeeze(1)
        context, weights = self.attention(hidden, encoder_states)
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

    return encoder, decoder, max_out_len


def greedy_decode(encoder, decoder, input_seq, max_out_len, vocab_size):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        x = torch.tensor([input_seq], dtype=torch.long)
        encoder_states, h_n = encoder(x)
        hidden = h_n.squeeze(0)
        step_input = torch.tensor([[START]], dtype=torch.long)
        output = []
        for _ in range(max_out_len):
            logits, hidden, _ = decoder.forward_step(step_input, hidden, encoder_states)
            next_token = logits[0].argmax().item()
            if next_token == END:
                break
            output.append(next_token)
            step_input = torch.tensor([[next_token]], dtype=torch.long)
    return output


def beam_search_decode(encoder, decoder, input_seq, max_out_len, vocab_size, beam_width=3):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        x = torch.tensor([input_seq], dtype=torch.long)
        encoder_states, h_n = encoder(x)
        init_hidden = h_n.squeeze(0)

        # each beam: (token_sequence, hidden_state, cumulative_log_prob, finished)
        beams = [([], init_hidden, 0.0, False)]

        for _ in range(max_out_len):
            candidates = []
            for seq, hidden, log_prob, finished in beams:
                if finished:
                    candidates.append((seq, hidden, log_prob, True))
                    continue
                last_token = seq[-1] if seq else START
                step_input = torch.tensor([[last_token]], dtype=torch.long)
                logits, new_hidden, _ = decoder.forward_step(step_input, hidden, encoder_states)
                log_probs = F.log_softmax(logits[0], dim=-1)

                top_log_probs, top_tokens = log_probs.topk(beam_width)
                for lp, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                    new_seq = seq + [tok]
                    new_finished = (tok == END)
                    candidates.append((new_seq, new_hidden, log_prob + lp, new_finished))

            # keep only the overall top `beam_width` candidates, by cumulative log-prob
            candidates.sort(key=lambda c: c[2], reverse=True)
            beams = candidates[:beam_width]

            if all(b[3] for b in beams):
                break

        best_seq = beams[0][0]
        if best_seq and best_seq[-1] == END:
            best_seq = best_seq[:-1]
        return best_seq


def demo_greedy_vs_beam():
    inputs, targets = make_reversal_dataset(n=600, min_len=9, max_len=12, vocab_size=8)
    print("=== Training attention seq2seq for greedy vs beam search comparison ===")
    encoder, decoder, max_out_len = train_attention_seq2seq(
        inputs, targets, n_epochs=180, hidden_size=32
    )

    test_inputs, test_targets = make_reversal_dataset(n=50, min_len=9, max_len=12, vocab_size=8, seed=88)

    greedy_correct = 0
    beam_correct = 0
    for inp, tgt in zip(test_inputs, test_targets):
        greedy_pred = greedy_decode(encoder, decoder, inp, max_out_len, vocab_size=8)
        beam_pred = beam_search_decode(encoder, decoder, inp, max_out_len, vocab_size=8, beam_width=8)
        if greedy_pred == tgt:
            greedy_correct += 1
        if beam_pred == tgt:
            beam_correct += 1

    print(f"\nGreedy decoding accuracy:      {greedy_correct}/{len(test_inputs)} = {greedy_correct/len(test_inputs):.2%}")
    print(f"Beam search (k=8) accuracy:    {beam_correct}/{len(test_inputs)} = {beam_correct/len(test_inputs):.2%}")
    print("\nBeam search explores multiple candidate sequences at each step instead of")
    print("committing greedily, which can recover from a locally-appealing-but-globally-")
    print("suboptimal early token choice that greedy decoding has no way to undo.")
    print("(On a harder, under-trained task like this deliberately-longer one, beam search's")
    print("advantage tends to show up more clearly than on an easy, fully-converged task.)")


if __name__ == "__main__":
    print("=== BLEU score ===")
    demo_bleu()

    print("=== Greedy vs beam search decoding ===")
    demo_greedy_vs_beam()
