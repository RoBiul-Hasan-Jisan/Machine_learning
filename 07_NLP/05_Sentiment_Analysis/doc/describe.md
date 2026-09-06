# 05. Sentiment Analysis

## Learning Objectives

- Build a sentiment classifier using classical (TF-IDF + logistic regression) and embedding-based approaches
- Explain the specific challenges sentiment analysis poses: negation, sarcasm, and domain dependence
- Evaluate a sentiment classifier with metrics appropriate for its class balance

## The Problem

Sentiment analysis — classifying text as positive, negative, or neutral — is one of the most common practical NLP tasks (product reviews, social media monitoring, customer support triage) and a natural first place to apply everything built up so far: text representation (Lesson 02), embeddings (Lessons 03-04), and eventually the classification architectures in Lesson 08. It's also a task with specific, well-known failure modes worth understanding directly, not just a generic classification exercise.

## The Concept

### Framing sentiment analysis as classification

At its simplest, sentiment analysis is many-to-one text classification (the RNN module's Lesson 01 terminology): an entire document maps to one label (positive/negative, or a finer-grained scale). The classical pipeline is:

```
Raw text -> Tokenize (Lesson 01) -> TF-IDF or embeddings (Lessons 02-04)
         -> Classifier (logistic regression, or an RNN/CNN, Lesson 08) -> Label
```

A TF-IDF + logistic regression baseline is a genuinely strong, fast starting point for many sentiment tasks and worth building before reaching for anything more complex — it's cheap to train, interpretable (you can inspect which words have the largest positive/negative coefficients), and a useful sanity check that the more sophisticated approaches in later lessons should meaningfully beat.

### Why sentiment analysis is harder than it looks

**Negation** flips meaning without changing the words present much: "not bad" is positive, "not good" is negative, but a plain bag-of-words representation (Lesson 02) sees "not," "bad" independently, losing the fact that "not" negates the word right after it. A simple partial fix: mark words following a negation word up to the next punctuation mark with a special prefix, so "not_bad" becomes its own distinct feature, giving a bag-of-words model at least a chance to learn that this combined token behaves differently than "bad" alone:

```python
def negate_tokens(tokens, negation_words={"not", "no", "never", "n't"}):
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
```

**Sarcasm and irony** are the hardest case: "Oh great, my flight got cancelled again" uses positive words ("great") to express negative sentiment, something no word-level feature (bag-of-words, TF-IDF, or even most embeddings) can reliably detect, since it requires understanding the gap between literal meaning and intended meaning — genuinely difficult even for large language models, and an active research problem rather than a solved one.

**Domain dependence**: word polarity isn't fixed across domains. "Unpredictable" is negative in a review of a car's handling but positive in a review of a movie's plot. A sentiment model trained on movie reviews will often perform noticeably worse when applied directly to product reviews or financial news, without some form of domain adaptation or retraining — a concrete instance of the distribution-shift problem the CNN and RNN modules both flagged as a common real-world failure mode.

**Comparative and mixed sentiment**: "This phone is better than my old one, but the battery life is disappointing" contains both positive and negative sentiment about different aspects of the same product. A single overall label loses this distinction — this is exactly what motivates **aspect-based sentiment analysis**, which assigns separate sentiment labels to different aspects (camera, battery, price) mentioned in the same review, rather than forcing one overall verdict.

### Evaluating a sentiment classifier

Sentiment datasets are frequently imbalanced (far more positive reviews than negative ones on many platforms, or vice versa for support tickets), which makes plain accuracy a misleading metric — a classifier that always predicts "positive" can achieve high accuracy on a heavily positive-skewed dataset while being useless. Precision, recall, and F1 per class (and a confusion matrix showing exactly which classes get confused with which) give a much more honest picture:

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
print(confusion_matrix(y_true, y_pred))
```

For a 3-class problem (positive/negative/neutral), it's especially worth checking whether errors cluster at the neutral boundary (predicting positive when the true label was neutral, or vice versa) rather than truly flipping polarity (predicting positive when the true label was negative) — the former is a much less severe error in most applications than the latter, a distinction plain accuracy doesn't capture at all.

See `code/sentiment_demo.py` for a complete TF-IDF + logistic regression sentiment classifier on a synthetic review dataset, the negation-handling preprocessing step applied and compared against the baseline, and a full classification report with a confusion matrix.

## Exercises

1. Build a TF-IDF + logistic regression sentiment classifier on a small labeled dataset (or the synthetic one in the code demo) and inspect the 10 words with the largest positive and negative coefficients. Confirm they make intuitive sense.
2. Implement `negate_tokens` and compare classifier accuracy with and without negation handling on a test set specifically containing several negated sentences (e.g. "not bad", "never disappointing").
3. Construct 5 sarcastic sentences with positive surface words but negative intended sentiment, and check how a TF-IDF-based classifier performs on them. Explain the failure in terms of what information the classifier does and doesn't have access to.
4. Using `classification_report` and a confusion matrix on an imbalanced synthetic dataset (e.g. 90% positive, 10% negative), compare accuracy against per-class F1 score, and explain why they tell different stories.

## Key Terms

| Term | What it actually means |
|---|---|
| Sentiment analysis | Classifying text by the sentiment (positive/negative/neutral, or a finer scale) it expresses |
| Negation handling | Preprocessing that marks words affected by a negation term, so a model can distinguish "bad" from "not bad" |
| Aspect-based sentiment analysis | Assigning separate sentiment labels to different aspects of the same subject mentioned in one piece of text, rather than one overall label |
| Domain dependence | The phenomenon where word polarity, and therefore model performance, shifts across different topic domains |
