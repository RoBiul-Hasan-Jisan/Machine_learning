"""
An NLI classifier using averaged word embeddings plus explicit
interaction features (difference, elementwise product), trained on a
small synthetic entailment/contradiction/neutral dataset.
"""

import numpy as np
import torch
import torch.nn as nn


LABELS = ["entailment", "contradiction", "neutral"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


def make_synthetic_nli_dataset(seed=0):
    """Hand-constructed premise/hypothesis/label triples built from
    templates, with enough repetition (via filler swaps) for training."""
    rng = np.random.default_rng(seed)

    subjects = ["a man", "a woman", "a person", "the athlete", "the musician"]
    actions_music = ["playing guitar", "singing a song", "playing piano"]
    entail_music = ["performing music", "making music", "playing an instrument"]
    contradict_music = ["staying completely silent", "not making any sound"]

    actions_sport = ["running fast", "playing soccer", "swimming laps"]
    entail_sport = ["exercising", "being physically active", "doing a sport"]
    contradict_sport = ["sitting still", "resting quietly"]

    neutral_facts = ["wearing a red shirt", "standing near a tree", "smiling brightly", "outdoors today"]

    examples = []
    for _ in range(60):
        subj = rng.choice(subjects)

        # entailment example
        action = rng.choice(actions_music)
        entail = rng.choice(entail_music)
        examples.append((f"{subj} is {action}", f"{subj.split()[-1]} is {entail}", "entailment"))

        # contradiction example
        contradict = rng.choice(contradict_music)
        examples.append((f"{subj} is {action}", f"{subj.split()[-1]} is {contradict}", "contradiction"))

        # neutral example
        fact = rng.choice(neutral_facts)
        examples.append((f"{subj} is {action}", f"{subj.split()[-1]} is {fact}", "neutral"))

        # sports variants too
        action2 = rng.choice(actions_sport)
        entail2 = rng.choice(entail_sport)
        examples.append((f"{subj} is {action2}", f"{subj.split()[-1]} is {entail2}", "entailment"))

        contradict2 = rng.choice(contradict_sport)
        examples.append((f"{subj} is {action2}", f"{subj.split()[-1]} is {contradict2}", "contradiction"))

        fact2 = rng.choice(neutral_facts)
        examples.append((f"{subj} is {action2}", f"{subj.split()[-1]} is {fact2}", "neutral"))

    return examples


def build_vocab(examples):
    vocab = {"<pad>": 0, "<unk>": 1}
    for premise, hypothesis, _ in examples:
        for tok in (premise + " " + hypothesis).lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


class NLIClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=24, hidden_size=32, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def encode(self, x):
        embedded = self.embedding(x)
        _, (h_n, c_n) = self.encoder(embedded)
        return h_n.squeeze(0)

    def forward(self, premise, hypothesis):
        p_vec = self.encode(premise)
        h_vec = self.encode(hypothesis)
        combined = torch.cat([p_vec, h_vec, torch.abs(p_vec - h_vec), p_vec * h_vec], dim=1)
        return self.classifier(combined)


def encode_text(text, vocab, max_len):
    ids = [vocab.get(w, 1) for w in text.lower().split()][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


def train_nli(examples, vocab, n_epochs=80, lr=0.01):
    max_len = max(len((p + " " + h).split()) for p, h, _ in examples)

    premises = torch.tensor([encode_text(p, vocab, max_len) for p, h, l in examples], dtype=torch.long)
    hypotheses = torch.tensor([encode_text(h, vocab, max_len) for p, h, l in examples], dtype=torch.long)
    labels = torch.tensor([LABEL_TO_IDX[l] for p, h, l in examples], dtype=torch.long)

    n = len(examples)
    perm = torch.randperm(n)
    premises, hypotheses, labels = premises[perm], hypotheses[perm], labels[perm]

    n_train = int(n * 0.8)
    p_train, p_test = premises[:n_train], premises[n_train:]
    h_train, h_test = hypotheses[:n_train], hypotheses[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    model = NLIClassifier(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(p_train, h_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(p_test, h_test)
                acc = (test_logits.argmax(1) == y_test).float().mean().item()
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  test_acc={acc:.3f}")

    return model, max_len, (p_test, h_test, y_test)


def demo_nli():
    examples = make_synthetic_nli_dataset()
    vocab = build_vocab(examples)
    print(f"Dataset size: {len(examples)}, vocab size: {len(vocab)}\n")

    model, max_len, (p_test, h_test, y_test) = train_nli(examples, vocab)

    print("\n=== Example predictions ===")
    model.eval()
    test_pairs = [
        ("a man is playing guitar", "man is performing music", "entailment"),
        ("a man is playing guitar", "man is staying completely silent", "contradiction"),
        ("a man is playing guitar", "man is wearing a red shirt", "neutral"),
        ("a woman is running fast", "woman is exercising", "entailment"),
        ("a woman is running fast", "woman is sitting still", "contradiction"),
    ]
    for premise, hypothesis, true_label in test_pairs:
        p_ids = torch.tensor([encode_text(premise, vocab, max_len)], dtype=torch.long)
        h_ids = torch.tensor([encode_text(hypothesis, vocab, max_len)], dtype=torch.long)
        with torch.no_grad():
            logits = model(p_ids, h_ids)
        pred_label = LABELS[logits.argmax(1).item()]
        marker = "OK" if pred_label == true_label else "WRONG"
        print(f"  Premise:    '{premise}'")
        print(f"  Hypothesis: '{hypothesis}'")
        print(f"  True: {true_label}   Predicted: {pred_label}  [{marker}]\n")


if __name__ == "__main__":
    demo_nli()
