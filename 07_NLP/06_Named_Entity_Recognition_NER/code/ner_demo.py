"""
A feature-based NER tagger (per-token logistic regression over
hand-engineered features) trained and evaluated on a small synthetic
dataset, including a demonstration of entity-level vs token-level
evaluation.
"""

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def token_features(tokens, i):
    token = tokens[i]
    return {
        "word": token.lower(),
        "is_capitalized": token[0].isupper(),
        "is_all_caps": token.isupper(),
        "prefix_2": token[:2].lower(),
        "suffix_2": token[-2:].lower(),
        "prev_word": tokens[i - 1].lower() if i > 0 else "<START>",
        "next_word": tokens[i + 1].lower() if i < len(tokens) - 1 else "<END>",
        "is_first_in_sentence": i == 0,
    }


def make_synthetic_ner_dataset():
    """(tokens, bio_labels) pairs, built from templates with swappable
    entity fillers so train/test genuinely differ (not exact duplicates)."""
    people = ["John", "Sarah", "Michael", "Emma", "David"]
    orgs = ["Apple", "Google", "Microsoft", "Amazon", "Tesla"]
    locs = ["Paris", "London", "Tokyo", "Berlin", "Madrid"]
    dates = ["Monday", "Tuesday", "Friday", "Wednesday"]

    templates = [
        (["{ORG}", "announced", "a", "new", "store", "in", "{LOC}", "on", "{DATE}"],
         ["B-ORG", "O", "O", "O", "O", "O", "B-LOC", "O", "B-DATE"]),
        (["{PER}", "visited", "{LOC}", "last", "{DATE}"],
         ["B-PER", "O", "B-LOC", "O", "B-DATE"]),
        (["The", "meeting", "with", "{ORG}", "is", "on", "{DATE}"],
         ["O", "O", "O", "B-ORG", "O", "O", "B-DATE"]),
        (["{PER}", "works", "at", "{ORG}"],
         ["B-PER", "O", "O", "B-ORG"]),
        (["The", "conference", "starts", "on", "{DATE}", "in", "{LOC}"],
         ["O", "O", "O", "O", "B-DATE", "O", "B-LOC"]),
        (["{ORG}", "and", "{ORG2}", "are", "based", "in", "{LOC}"],
         ["B-ORG", "O", "B-ORG", "O", "O", "O", "B-LOC"]),
    ]

    rng = np.random.default_rng(0)
    data = []
    for _ in range(90):
        template_tokens, template_labels = templates[rng.integers(len(templates))]
        person = rng.choice(people)
        org = rng.choice(orgs)
        org2 = rng.choice([o for o in orgs if o != org])
        loc = rng.choice(locs)
        date = rng.choice(dates)

        tokens = [t.format(PER=person, ORG=org, ORG2=org2, LOC=loc, DATE=date) for t in template_tokens]
        data.append((tokens, list(template_labels)))

    return data


def extract_entities(tokens, labels):
    """Convert a BIO-tagged sequence into a set of (start, end, type) entity spans."""
    entities = []
    current_type = None
    current_start = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_type is not None:
                entities.append((current_start, i, current_type))
            current_type = label[2:]
            current_start = i
        elif label.startswith("I-") and current_type == label[2:]:
            continue  # extend current entity
        else:  # "O", or an I- that doesn't match the current open entity
            if current_type is not None:
                entities.append((current_start, i, current_type))
            current_type = None
    if current_type is not None:
        entities.append((current_start, len(labels), current_type))
    return set(entities)


def entity_level_prf(true_labels_list, pred_labels_list):
    tp, fp, fn = 0, 0, 0
    for true_labels, pred_labels in zip(true_labels_list, pred_labels_list):
        true_entities = extract_entities(range(len(true_labels)), true_labels)
        pred_entities = extract_entities(range(len(pred_labels)), pred_labels)
        tp += len(true_entities & pred_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def demo_ner_tagger():
    data = make_synthetic_ner_dataset()
    rng = np.random.default_rng(1)
    rng.shuffle(data)

    # Flatten into per-token (features, label) pairs for training,
    # but keep sentence structure for entity-level evaluation
    all_features, all_labels = [], []
    sentence_boundaries = []
    for tokens, labels in data:
        start = len(all_features)
        for i in range(len(tokens)):
            all_features.append(token_features(tokens, i))
            all_labels.append(labels[i])
        sentence_boundaries.append((start, len(all_features)))

    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(all_features)

    n_train_sentences = int(len(data) * 0.75)
    train_end_token = sentence_boundaries[n_train_sentences - 1][1]

    X_train, X_test = X[:train_end_token], X[train_end_token:]
    y_train, y_test = all_labels[:train_end_token], all_labels[train_end_token:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    token_acc = accuracy_score(y_test, y_pred)
    print(f"Token-level accuracy: {token_acc:.4f}")

    # Reconstruct per-sentence label sequences for entity-level eval
    test_sentences = data[n_train_sentences:]
    true_seqs, pred_seqs = [], []
    offset = 0
    for tokens, true_labels in test_sentences:
        n = len(tokens)
        pred_seqs.append(list(y_pred[offset:offset + n]))
        true_seqs.append(true_labels)
        offset += n

    precision, recall, f1 = entity_level_prf(true_seqs, pred_seqs)
    print(f"Entity-level precision: {precision:.4f}")
    print(f"Entity-level recall:    {recall:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n=== Example prediction ===")
    tokens, true_labels = test_sentences[0]
    pred_labels = pred_seqs[0]
    print(f"{'token':12s} | {'true':8s} | {'predicted':8s}")
    for t, true_l, pred_l in zip(tokens, true_labels, pred_labels):
        marker = "" if true_l == pred_l else "  <-- mismatch"
        print(f"{t:12s} | {true_l:8s} | {pred_l:8s}{marker}")


def demo_o_dominant_baseline():
    """Show that a trivial 'always predict O' baseline scores deceptively
    high on TOKEN accuracy, motivating entity-level evaluation."""
    data = make_synthetic_ner_dataset()
    all_true_labels = [label for _, labels in data for label in labels]
    o_baseline_preds = ["O"] * len(all_true_labels)

    token_acc = accuracy_score(all_true_labels, o_baseline_preds)
    print(f"\n'Always predict O' baseline token-level accuracy: {token_acc:.4f}")

    true_seqs = [labels for _, labels in data]
    pred_seqs = [["O"] * len(labels) for labels in true_seqs]
    precision, recall, f1 = entity_level_prf(true_seqs, pred_seqs)
    print(f"'Always predict O' baseline entity-level F1: {f1:.4f}")
    print("\nHigh token accuracy, zero entity-level performance -- exactly why")
    print("NER must be evaluated at the entity level, not the token level.")


if __name__ == "__main__":
    print("=== Feature-based NER tagger ===")
    demo_ner_tagger()

    print("\n=== Why token accuracy is misleading for NER ===")
    demo_o_dominant_baseline()
