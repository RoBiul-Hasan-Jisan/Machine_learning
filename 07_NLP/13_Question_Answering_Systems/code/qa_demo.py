"""
A from-scratch span-extraction QA model (bidirectional RNN encoder +
start/end prediction head) trained on a small synthetic dataset, plus a
minimal retriever-reader pipeline combining TF-IDF retrieval with the
trained extractive reader.
"""

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def make_synthetic_qa_dataset(seed=0):
    rng = np.random.default_rng(seed)

    subjects = ["the tower", "the bridge", "the museum", "the stadium", "the library"]
    heights = ["50 meters", "120 meters", "80 meters", "200 meters", "35 meters"]
    years = ["1889", "1932", "1965", "1998", "2004"]

    examples = []
    for _ in range(300):
        subj = rng.choice(subjects)
        height = rng.choice(heights)
        year = rng.choice(years)

        passage = f"{subj} was completed in {year} and stands {height} tall"
        passage_tokens = tokenize(passage)

        if rng.random() < 0.5:
            question = f"how tall is {subj}"
            answer_tokens = height.split()
        else:
            question = f"when was {subj} completed"
            answer_tokens = [year]

        question_tokens = tokenize(question)

        ans_len = len(answer_tokens)
        start_idx = None
        for i in range(len(passage_tokens) - ans_len + 1):
            if passage_tokens[i:i + ans_len] == answer_tokens:
                start_idx = i
                break
        if start_idx is None:
            continue
        end_idx = start_idx + ans_len - 1

        examples.append((passage_tokens, question_tokens, start_idx, end_idx))

    return examples


def build_vocab(examples):
    vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    for passage, question, _, _ in examples:
        for tok in passage + question:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


class SpanQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.start_classifier = nn.Linear(hidden_size * 2, 1)
        self.end_classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, combined_ids, passage_len):
        embedded = self.embedding(combined_ids)
        encoded, _ = self.encoder(embedded)
        passage_encoded = encoded[:, :passage_len, :]
        start_logits = self.start_classifier(passage_encoded).squeeze(-1)
        end_logits = self.end_classifier(passage_encoded).squeeze(-1)
        return start_logits, end_logits


def encode_example(passage, question, vocab, max_passage_len):
    passage_ids = [vocab.get(t, 1) for t in passage][:max_passage_len]
    passage_ids += [0] * (max_passage_len - len(passage_ids))
    question_ids = [vocab.get(t, 1) for t in question]
    combined = passage_ids + [vocab["<sep>"]] + question_ids
    return combined, max_passage_len


def train_qa_model(examples, vocab, n_epochs=60, lr=0.01):
    max_passage_len = max(len(p) for p, _, _, _ in examples)
    max_total_len = max_passage_len + 1 + max(len(q) for _, q, _, _ in examples)

    all_combined, all_starts, all_ends = [], [], []
    for passage, question, start, end in examples:
        combined, _ = encode_example(passage, question, vocab, max_passage_len)
        padded_combined = combined + [0] * (max_total_len - len(combined))
        all_combined.append(padded_combined)
        all_starts.append(start)
        all_ends.append(end)

    X = torch.tensor(all_combined, dtype=torch.long)
    y_start = torch.tensor(all_starts, dtype=torch.long)
    y_end = torch.tensor(all_ends, dtype=torch.long)

    model = SpanQAModel(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_train = int(len(examples) * 0.85)
    X_train, X_test = X[:n_train], X[n_train:]
    y_start_train, y_start_test = y_start[:n_train], y_start[n_train:]
    y_end_train, y_end_test = y_end[:n_train], y_end[n_train:]

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        start_logits, end_logits = model(X_train, max_passage_len)
        loss = loss_fn(start_logits, y_start_train) + loss_fn(end_logits, y_end_train)
        loss.backward()
        optimizer.step()

        if epoch % 15 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                s_logits, e_logits = model(X_test, max_passage_len)
                start_acc = (s_logits.argmax(1) == y_start_test).float().mean().item()
                end_acc = (e_logits.argmax(1) == y_end_test).float().mean().item()
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  start_acc={start_acc:.3f}  end_acc={end_acc:.3f}")

    return model, max_passage_len


def demo_span_qa():
    examples = make_synthetic_qa_dataset()
    vocab = build_vocab(examples)
    print(f"Vocab size: {len(vocab)}, dataset size: {len(examples)}\n")

    model, max_passage_len = train_qa_model(examples, vocab)

    print("\n=== Example predictions ===")
    model.eval()
    for passage, question, true_start, true_end in examples[-5:]:
        combined, _ = encode_example(passage, question, vocab, max_passage_len)
        x = torch.tensor([combined], dtype=torch.long)
        with torch.no_grad():
            start_logits, end_logits = model(x, max_passage_len)
        pred_start = start_logits[0].argmax().item()
        pred_end = end_logits[0].argmax().item()

        true_answer = " ".join(passage[true_start:true_end + 1])
        pred_answer = " ".join(passage[pred_start:pred_end + 1]) if pred_end >= pred_start else "(invalid span)"

        print(f"Passage:  {' '.join(passage)}")
        print(f"Question: {' '.join(question)}")
        print(f"True answer: '{true_answer}'   Predicted: '{pred_answer}'\n")

    return model, vocab, max_passage_len


def tfidf_retrieve(query, passages, top_k=2):
    tokenized_docs = [tokenize(p) for p in passages]
    vocab = sorted(set(w for doc in tokenized_docs for w in doc) | set(tokenize(query)))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    N = len(passages)
    df = Counter()
    for doc in tokenized_docs:
        for w in set(doc):
            df[w] += 1
    idf = {w: np.log((N + 1) / (1 + df[w])) + 1 for w in vocab}

    def vectorize(tokens):
        v = np.zeros(len(vocab))
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        for w, c in counts.items():
            if w in word_to_idx:
                v[word_to_idx[w]] = (c / total) * idf[w]
        return v

    doc_vectors = np.array([vectorize(doc) for doc in tokenized_docs])
    query_vector = vectorize(tokenize(query))

    norms = np.linalg.norm(doc_vectors, axis=1) * (np.linalg.norm(query_vector) + 1e-10)
    norms[norms == 0] = 1e-10
    sims = (doc_vectors @ query_vector) / norms

    top_idx = np.argsort(-sims)[:top_k]
    return [(passages[i], sims[i]) for i in top_idx]


def demo_retriever_reader(model, vocab, max_passage_len):
    print("=== Minimal retriever-reader pipeline ===")
    passages = [
        "the tower was completed in 1889 and stands 50 meters tall",
        "the bridge was completed in 1932 and stands 120 meters tall",
        "the museum was completed in 1965 and stands 80 meters tall",
        "the library was completed in 2004 and stands 35 meters tall",
    ]
    question = "how tall is the museum"

    retrieved = tfidf_retrieve(question, passages, top_k=2)
    print(f"Question: '{question}'")
    print("Top retrieved passages:")
    for p, score in retrieved:
        print(f"  [{score:.4f}] {p}")

    best_passage = retrieved[0][0]
    passage_tokens = tokenize(best_passage)
    question_tokens = tokenize(question)
    combined, _ = encode_example(passage_tokens, question_tokens, vocab, max_passage_len)
    x = torch.tensor([combined], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        start_logits, end_logits = model(x, max_passage_len)
    pred_start = start_logits[0].argmax().item()
    pred_end = end_logits[0].argmax().item()
    answer = " ".join(passage_tokens[pred_start:pred_end + 1]) if pred_end >= pred_start else "(invalid span)"

    print(f"\nReader extracts from the top passage: '{answer}'")


if __name__ == "__main__":
    model, vocab, max_passage_len = demo_span_qa()
    demo_retriever_reader(model, vocab, max_passage_len)
