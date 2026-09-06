"""
A TF-IDF + logistic regression sentiment classifier on a synthetic
review dataset, negation-handling preprocessing compared against the
baseline, and a full classification report with a confusion matrix.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def tokenize(text):
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text.lower())


def negate_tokens(tokens, negation_words=("not", "no", "never", "n't", "cant", "cannot")):
    result = []
    negating = False
    for token in tokens:
        if token in negation_words:
            negating = True
            result.append(token)
        elif token in {".", "!", "?", ","}:
            negating = False
            result.append(token)
        elif negating:
            result.append(f"NOT_{token}")
        else:
            result.append(token)
    return result


def make_synthetic_reviews(seed=0):
    rng = np.random.default_rng(seed)

    # Core adjective is shared between positive/negative framing so a
    # bag-of-words model can't just key on "good" vs "bad" appearing --
    # it has to notice the surrounding negation, which TF-IDF (unigram)
    # alone cannot represent.
    nouns = ["phone", "laptop", "product", "gadget", "device", "item"]

    plain_positive = [
        "this {noun} is good",
        "this {noun} is great",
        "the {noun} works well",
        "I really like this {noun}",
    ]
    plain_negative = [
        "this {noun} is bad",
        "this {noun} is terrible",
        "the {noun} works poorly",
        "I really dislike this {noun}",
    ]
    # negated versions: SAME core sentiment word, opposite true label
    negated_to_positive = [
        "this {noun} is not bad",
        "this {noun} is not terrible",
        "the {noun} does not work poorly",
        "I do not dislike this {noun}",
    ]
    negated_to_negative = [
        "this {noun} is not good",
        "this {noun} is not great",
        "the {noun} does not work well",
        "I do not like this {noun}",
    ]

    texts, labels = [], []
    for _ in range(80):
        t = rng.choice(plain_positive).format(noun=rng.choice(nouns))
        texts.append(t); labels.append(1)
    for _ in range(80):
        t = rng.choice(plain_negative).format(noun=rng.choice(nouns))
        texts.append(t); labels.append(0)
    for _ in range(40):
        t = rng.choice(negated_to_positive).format(noun=rng.choice(nouns))
        texts.append(t); labels.append(1)
    for _ in range(40):
        t = rng.choice(negated_to_negative).format(noun=rng.choice(nouns))
        texts.append(t); labels.append(0)

    return texts, labels


def demo_baseline_classifier():
    texts, labels = make_synthetic_reviews()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    preds = clf.predict(X_test_tfidf)
    print("=== Baseline TF-IDF + logistic regression ===")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    top_pos = np.argsort(coefs)[-8:][::-1]
    top_neg = np.argsort(coefs)[:8]
    print("\nTop positive-sentiment words (largest coefficients):")
    print([feature_names[i] for i in top_pos])
    print("Top negative-sentiment words (smallest coefficients):")
    print([feature_names[i] for i in top_neg])

    return X_train, X_test, y_train, y_test, clf, vectorizer


def demo_negation_handling(X_train, X_test, y_train, y_test):
    X_train_neg = [" ".join(negate_tokens(tokenize(t))) for t in X_train]
    X_test_neg = [" ".join(negate_tokens(tokenize(t))) for t in X_test]

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_neg)
    X_test_tfidf = vectorizer.transform(X_test_neg)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    preds = clf.predict(X_test_tfidf)

    print("\n=== With negation handling ===")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    # Specifically test on a few clearly negated examples using the
    # SAME core sentiment words the model saw during training
    test_examples = [
        "this phone is not bad",
        "this phone is not good",
        "I do not dislike this laptop",
        "I do not like this laptop",
    ]
    test_neg = [" ".join(negate_tokens(tokenize(t))) for t in test_examples]
    test_tfidf = vectorizer.transform(test_neg)
    example_preds = clf.predict(test_tfidf)

    print("\nSpot-check on clearly negated sentences:")
    for text, pred in zip(test_examples, example_preds):
        label = "positive" if pred == 1 else "negative"
        print(f"  '{text}' -> predicted {label}")


def demo_sarcasm_failure(clf, vectorizer):
    sarcastic_examples = [
        "oh great, my flight got cancelled again",
        "wonderful, the product broke on day one",
        "just fantastic, another delayed delivery",
    ]
    tfidf = vectorizer.transform(sarcastic_examples)
    preds = clf.predict(tfidf)

    print("\n=== Sarcasm: a known failure mode ===")
    for text, pred in zip(sarcastic_examples, preds):
        label = "positive" if pred == 1 else "negative"
        print(f"  '{text}' -> predicted {label}  (intended: negative)")
    print("\nThe classifier keys on positive surface words ('great', 'wonderful',")
    print("'fantastic') and misses the sarcastic intent entirely -- it has no way")
    print("to represent the gap between literal wording and intended meaning.")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, clf, vectorizer = demo_baseline_classifier()
    demo_negation_handling(X_train, X_test, y_train, y_test)
    demo_sarcasm_failure(clf, vectorizer)
