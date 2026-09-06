"""
Contrastive sentence embedding training from scratch on synthetic
anchor/positive/negative triples, plus a simplified bi-encoder vs
cross-encoder comparison illustrating the speed/accuracy tradeoff.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_contrastive_triples(seed=0):
    """(anchor, positive, negative) sentence triples, built so anchor/positive
    share meaning (different surface words) and anchor/negative don't."""
    rng = np.random.default_rng(seed)

    music_anchor = ["a man is playing guitar", "a woman is singing a song", "someone is playing piano"]
    music_positive = ["a person is performing music", "someone is making music", "a musician is playing an instrument"]

    sport_anchor = ["a man is running fast", "a woman is playing soccer", "someone is swimming laps"]
    sport_positive = ["a person is exercising", "someone is being active", "an athlete is doing a sport"]

    unrelated = ["the weather is sunny today", "the store closes at five", "the book is on the table",
                 "the coffee is too hot", "the car needs gas"]

    triples = []
    for _ in range(80):
        if rng.random() < 0.5:
            anchor = rng.choice(music_anchor)
            positive = rng.choice(music_positive)
        else:
            anchor = rng.choice(sport_anchor)
            positive = rng.choice(sport_positive)
        negative = rng.choice(unrelated)
        triples.append((anchor, positive, negative))

    return triples


def build_vocab(triples):
    vocab = {"<pad>": 0, "<unk>": 1}
    for a, p, n in triples:
        for text in (a, p, n):
            for tok in text.lower().split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def encode_text(text, vocab, max_len):
    ids = [vocab.get(w, 1) for w in text.lower().split()][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=24, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, c_n) = self.rnn(embedded)
        return F.normalize(h_n.squeeze(0), dim=-1)  # normalized -> cosine sim = dot product


def contrastive_loss(anchor_vec, positive_vec, negative_vec, temperature=0.1):
    """InfoNCE-style loss: pull anchor/positive together, push anchor/negative apart."""
    pos_sim = (anchor_vec * positive_vec).sum(dim=-1) / temperature
    neg_sim = (anchor_vec * negative_vec).sum(dim=-1) / temperature

    logits = torch.stack([pos_sim, neg_sim], dim=1)  # (batch, 2) -- positive is "class 0"
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    return F.cross_entropy(logits, labels)


def train_sentence_encoder(triples, vocab, n_epochs=100, lr=0.01):
    max_len = max(len(t.split()) for triple in triples for t in triple)

    anchors = torch.tensor([encode_text(a, vocab, max_len) for a, p, n in triples], dtype=torch.long)
    positives = torch.tensor([encode_text(p, vocab, max_len) for a, p, n in triples], dtype=torch.long)
    negatives = torch.tensor([encode_text(n, vocab, max_len) for a, p, n in triples], dtype=torch.long)

    model = SentenceEncoder(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        a_vec = model(anchors)
        p_vec = model(positives)
        n_vec = model(negatives)
        loss = contrastive_loss(a_vec, p_vec, n_vec)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                pos_sim = (a_vec * p_vec).sum(-1).mean().item()
                neg_sim = (a_vec * n_vec).sum(-1).mean().item()
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  avg_pos_sim={pos_sim:.3f}  avg_neg_sim={neg_sim:.3f}")

    return model, max_len


def average_word_embeddings(text, embedding_table, vocab):
    ids = [vocab.get(w, 1) for w in text.lower().split()]
    vecs = embedding_table[ids]
    return vecs.mean(dim=0)


def demo_contrastive_training():
    triples = make_contrastive_triples()
    vocab = build_vocab(triples)
    print(f"Dataset: {len(triples)} triples, vocab size: {len(vocab)}\n")

    model, max_len = train_sentence_encoder(triples, vocab)

    print("\n=== Testing on held-out sentence pairs ===")
    model.eval()
    test_cases = [
        ("a man is playing guitar", "a person is performing music", "should be SIMILAR (paraphrase)"),
        ("a man is playing guitar", "the weather is sunny today", "should be DIFFERENT (unrelated)"),
        ("the dog bit the man", "the man bit the dog", "same words, opposite meaning -- a genuine test"),
    ]
    for text_a, text_b, note in test_cases:
        a_ids = torch.tensor([encode_text(text_a, vocab, max_len)], dtype=torch.long)
        b_ids = torch.tensor([encode_text(text_b, vocab, max_len)], dtype=torch.long)
        with torch.no_grad():
            a_vec = model(a_ids)
            b_vec = model(b_ids)
        sim = (a_vec * b_vec).sum().item()
        print(f"  '{text_a}' vs '{text_b}'")
        print(f"    cosine similarity = {sim:.4f}   ({note})\n")

    print("Note on the reordering pair: this toy model was only ever trained on")
    print("paraphrase-vs-unrelated contrasts, never specifically on word-order-flips,")
    print("so its similarity score for 'dog bit man' vs 'man bit dog' stays fairly high")
    print("-- the LSTM encoder CAN in principle represent order (unlike averaged word")
    print("embeddings, which would score this pair as IDENTICAL), but contrastive")
    print("training only teaches whatever distinctions its training pairs actually")
    print("contain. Real sentence embedding models train on much larger, more varied")
    print("data specifically including this kind of hard negative.")


def demo_biencoder_vs_crossencoder_speed():
    """Illustrate the core speed tradeoff: pre-computed bi-encoder vectors
    vs a cross-encoder needing a fresh forward pass per pair."""
    n_documents = 200
    embedding_dim = 32

    rng = np.random.default_rng(0)
    doc_vectors = torch.tensor(rng.normal(size=(n_documents, embedding_dim)), dtype=torch.float32)
    doc_vectors = F.normalize(doc_vectors, dim=-1)
    query_vector = F.normalize(torch.tensor(rng.normal(size=embedding_dim), dtype=torch.float32), dim=0)

    # Bi-encoder: documents are ALREADY encoded (pre-computed) -- just compare
    start = time.perf_counter()
    bi_encoder_scores = doc_vectors @ query_vector
    bi_encoder_time = time.perf_counter() - start

    # Cross-encoder: simulate a fresh small "joint" forward pass PER document
    cross_encoder_net = nn.Sequential(nn.Linear(embedding_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1))
    start = time.perf_counter()
    cross_encoder_scores = []
    with torch.no_grad():
        for i in range(n_documents):
            combined = torch.cat([query_vector, doc_vectors[i]])
            score = cross_encoder_net(combined)
            cross_encoder_scores.append(score.item())
    cross_encoder_time = time.perf_counter() - start

    print(f"Scoring {n_documents} documents against 1 query:")
    print(f"  Bi-encoder (pre-computed vectors, cheap comparison):  {bi_encoder_time*1000:.3f} ms")
    print(f"  Cross-encoder (fresh joint forward pass per doc):     {cross_encoder_time*1000:.3f} ms")
    print(f"  Cross-encoder is {cross_encoder_time / bi_encoder_time:.0f}x slower in this toy comparison")
    print("\n(This gap grows dramatically with a real Transformer-sized cross-encoder")
    print("and a real document collection -- exactly why production systems retrieve")
    print("broadly with a bi-encoder, then re-rank only a small candidate set with a")
    print("cross-encoder, rather than cross-encoding the entire collection per query.)")


if __name__ == "__main__":
    print("=== Contrastive sentence embedding training ===")
    demo_contrastive_training()

    print("=== Bi-encoder vs cross-encoder speed ===")
    demo_biencoder_vs_crossencoder_speed()
