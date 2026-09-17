# Naive Bayes 
---



Naive Bayes is a **classification algorithm** — given some input features, it predicts which category (class) something belongs to.

Common real-world uses:

- **Spam detection** — is this email spam or not?
- **Sentiment analysis** — is this review positive or negative?
- **Document categorization** — is this article about sports, politics, or tech?
- **Medical diagnosis** — does a patient likely have a condition, given symptoms?

It is called a **probabilistic** classifier because instead of just outputting a label, it computes the *probability* of each possible class and picks the most likely one.

It is called **naive** because it makes one simplifying (and technically incorrect) assumption: **all features are independent of each other, given the class.** More on why this doesn't ruin the algorithm in Level 9.

---

##  The Core Idea: Bayes' Theorem

Everything in this algorithm rests on one equation, Bayes' Theorem:

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
$$

In plain language:

$$
\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}
$$

| Symbol | Name | Meaning |
|--------|------|---------|
| P(A) | Prior | What we believed *before* seeing evidence |
| P(B\|A) | Likelihood | How probable the evidence is, if A is true |
| P(A\|B) | Posterior | What we believe *after* seeing evidence |
| P(B) | Evidence | Overall probability of observing that evidence |

### Intuition Example

Say 1% of a population has a disease, and a test is 90% accurate. If someone tests positive, Bayes' Theorem tells you the *actual* probability they have the disease — which is usually much lower than 90%, because false positives on the healthy 99% add up. This "updating a belief given evidence" is exactly what Naive Bayes does for classification.

---

## Turning Bayes' Theorem Into a Classifier

We want to know: given features $X = (x_1, x_2, \ldots, x_n)$, what is the most likely class $C$?

$$
P(C \mid X) = \frac{P(X \mid C)\, P(C)}{P(X)}
$$

Since $P(X)$ is the same regardless of which class we're testing, we can ignore it and just compare:

$$
P(X \mid C)\, P(C)
$$

across all candidate classes — whichever is largest wins.

### The Independence Assumption

Computing $P(X \mid C) = P(x_1, x_2, \ldots, x_n \mid C)$ directly is expensive and requires huge amounts of data. Naive Bayes sidesteps this by assuming each feature is independent given the class:

$$
P(X \mid C) = P(x_1 \mid C)\, P(x_2 \mid C) \cdots P(x_n \mid C)
$$

### The Final Formula

$$
P(C \mid X) \;\propto\; P(C) \prod_{i=1}^{n} P(x_i \mid C)
$$

This is the entire algorithm. Everything else is detail.

---

##  Worked Example by Hand

**Training data:**

| Email | Spam? |
|-------|-------|
| FREE MONEY | Yes |
| WIN MONEY | Yes |
| MEETING TODAY | No |
| PROJECT UPDATE | No |

**Step 1 — Priors:**

$$
P(\text{Spam}) = \frac{2}{4} = 0.5 \qquad P(\text{Not Spam}) = \frac{2}{4} = 0.5
$$

**Step 2 — Likelihoods:** "FREE" appears in 1 of the 2 spam emails, and 0 of the 2 non-spam emails:

$$
P(\text{FREE} \mid \text{Spam}) = \frac{1}{2}, \qquad P(\text{FREE} \mid \text{Not Spam}) = 0
$$

**Step 3 — Classify a new email:** "FREE WIN"

Multiply the prior by each word's likelihood under each class:

$$
\text{Score}(\text{Spam}) = P(\text{Spam}) \times P(\text{FREE}\mid\text{Spam}) \times P(\text{WIN}\mid\text{Spam})
$$

$$
\text{Score}(\text{Not Spam}) = P(\text{Not Spam}) \times P(\text{FREE}\mid\text{Not Spam}) \times P(\text{WIN}\mid\text{Not Spam})
$$

Whichever score is higher determines the predicted class. (Note: with a raw zero probability here, we'd need Laplace smoothing — see Level 6 — before this works cleanly.)

---

## The Three Variants (and When to Use Each)

The right variant depends entirely on what your features look like.

### Gaussian Naive Bayes
- **Use for:** continuous numeric data (height, weight, age, income)
- **Assumption:** each feature follows a normal (bell curve) distribution per class
- **Stores:** mean (μ) and variance (σ²) per feature, per class

$$
P(x \mid C) = \frac{1}{\sqrt{2\pi\sigma^2}} \, e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

### Multinomial Naive Bayes
- **Use for:** text classification — the most common real-world application
- **Assumption:** features are word/term counts
- **Example:** "free" appears 50 times, "money" appears 30 times → probabilities derived from frequency

### Bernoulli Naive Bayes
- **Use for:** binary features (presence/absence, not frequency)
- **Example:** "Does the email contain the word FREE?" → 0 or 1

| Data type | Variant |
|-----------|---------|
| Continuous numbers | Gaussian |
| Word counts / text frequency | Multinomial |
| Binary yes/no features | Bernoulli |

---

##  Laplace Smoothing (Fixing the Zero-Probability Problem)

**The problem:** If a word never appears in the training data for a class, its probability is exactly 0. Because probabilities are *multiplied* together, a single zero makes the entire product zero — no matter how strong the other evidence is.

**The fix — add 1 to every count:**

$$
P(x_i \mid C) = \frac{\text{count}(x_i, C) + 1}{\text{total}(C) + V}
$$

where $V$ is the vocabulary size (total number of distinct features). This is called **Laplace smoothing** (or **add-one smoothing**), and virtually every production implementation uses some form of it.

---

##  Implementation in Python

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Continuous features (e.g. sensor readings, measurements)
model = GaussianNB()
model.fit(X_train, y_train)

# Word-count / text features
model = MultinomialNB(alpha=1.0)   # alpha = Laplace smoothing strength
model.fit(X_train, y_train)

# Binary presence/absence features
model = BernoulliNB(alpha=1.0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # class probabilities, not just labels
```

A typical text-classification pipeline in practice:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("vectorizer", CountVectorizer(stop_words="english")),
    ("classifier", MultinomialNB(alpha=1.0)),
])

pipeline.fit(train_texts, train_labels)
predicted = pipeline.predict(new_texts)
```

---

## Numerical Stability With Log-Probabilities

Multiplying many small probabilities together causes **underflow** — the result becomes so small that it rounds to zero on a computer, even before smoothing is considered. Production implementations therefore work in **log-space**:

$$
\log P(C \mid X) \;\propto\; \log P(C) + \sum_{i=1}^{n} \log P(x_i \mid C)
$$

Turning products into sums avoids underflow entirely, and this is exactly what `scikit-learn` and virtually every serious implementation does internally. If you ever implement Naive Bayes from scratch, **always sum log-probabilities rather than multiplying raw probabilities.**

---

##  Why It Works Despite a False Assumption

The independence assumption is almost never literally true — in a spam email, "FREE" and "MONEY" are correlated, not independent. So why does Naive Bayes still perform well?

The key insight: **Naive Bayes only needs to rank classes correctly, not estimate calibrated probabilities correctly.**

Correlations between features tend to bias the probability estimates for *every* class in a similar direction. Since classification only cares about *which* class scores highest — not the exact probability value — these biases often cancel out when comparing classes, even though the absolute probability outputs are distorted. This is why Naive Bayes can make the right classification decision while simultaneously producing poorly calibrated confidence scores.

**Practical implication:** trust Naive Bayes for the predicted label, but be cautious about treating its `predict_proba()` output as a true probability — calibrate it (e.g., with `CalibratedClassifierCV`) if you need accurate confidence values.

---

##  Limitations, Failure Modes, and When to Move On

| Limitation | Why it matters |
|------------|-----------------|
| Independence assumption | Breaks down badly when features are strongly correlated (e.g., highly redundant text features) |
| No feature interactions | Cannot learn "feature A matters only when feature B is also true" |
| Poor probability calibration | Confidence scores are often overconfident or skewed |
| Continuous-data assumption | Gaussian NB assumes normality; badly skewed data hurts performance |
| Ceiling on accuracy | On complex tabular data, gradient boosting (e.g., XGBoost) usually wins |

**When to prefer something else:**
- If features have strong, meaningful interactions → try logistic regression, tree ensembles, or gradient boosting.
- If you need well-calibrated probabilities → calibrate NB's output, or switch to logistic regression.
- If you have abundant data and complex tabular structure → boosting methods typically outperform.

**When Naive Bayes is genuinely the right call:**
- Very high-dimensional, sparse data (classic text classification)
- Small training sets where more complex models would overfit
- You need a fast baseline before investing in something heavier
- Latency or memory constraints matter — NB is close to unbeatable on speed

---

## Quick-Reference Summary

**The algorithm in one line:**
Bayes' Theorem + "features are independent given the class" → fast, simple probabilistic classification.

**The formula:**

$$
P(C \mid X) \;\propto\; P(C) \prod_{i=1}^{n} P(x_i \mid C)
$$

**Choosing a variant:**

```
Text / word counts   → Multinomial Naive Bayes
Continuous numbers   → Gaussian Naive Bayes
Binary features      → Bernoulli Naive Bayes
```

**Don't forget:**
- Apply Laplace smoothing to avoid zero probabilities
- Use log-probabilities in any real implementation to avoid numerical underflow
- Trust the predicted class more than the raw predicted probability
