# 06 · Practical Deep Learning

You can now build, train, optimize, regularize, and correctly initialize a network. This final lesson covers the methodology around all of that: how to structure your data, diagnose what's actually wrong with a model, search for good hyperparameters, and correctly measure whether your model is actually good.

## What you'll learn
- **Train / validation / test split** — and why you need all three, not just two
- **Underfitting vs. overfitting** — the two failure modes, and how to tell them apart
- **Bias vs. variance** — the formal framework underneath under/overfitting
- **Hyperparameter tuning** — how to systematically search for good settings
- **Evaluation metrics** — why accuracy alone is often misleading
- **Confusion matrix** — the foundation every classification metric is built from
- **Precision / Recall / F1** — metrics that matter when classes are imbalanced or errors aren't equally costly
- **ROC-AUC** — evaluating a classifier across every possible decision threshold at once

---

## 1. Train / Validation / Test Split

Three separate datasets, three separate jobs:

- **Training set** — what the model's weights are actually fit to (forward/backward pass, gradient updates).
- **Validation set** — used *during development* to make decisions: which hyperparameters, which architecture, when to early-stop. The model never trains directly on this data, but you (the developer) *do* make decisions based on it — so it's not a fully unbiased final measure.
- **Test set** — touched exactly once, at the very end, to report a final, honest performance number. If you tune anything based on test-set performance, it silently becomes a second validation set, and your final reported number stops being trustworthy.

A common split is 60/20/20 or 70/15/15 (train/val/test), though the right ratio depends on how much data you have — with very large datasets, even a 98/1/1 split can leave the val/test sets large enough to be statistically reliable.

## 2. Underfitting vs. Overfitting

- **Underfitting:** the model is too simple (or undertrained) to capture the true pattern in the data. Symptom: both training AND validation error are high, and close to each other.
- **Overfitting:** the model has learned the training data's noise/specifics rather than the general pattern. Symptom: training error is low, but validation error is much higher — a large train/val **gap** (exactly what lesson 04's regularization techniques target).
- **Good fit:** both training and validation error are low, and reasonably close to each other.

## 3. Bias vs. Variance

The formal decomposition underneath under/overfitting. A model's expected test error can be decomposed as:

```
Expected Error = Bias² + Variance + Irreducible Noise
```

- **Bias** — error from the model being too simple to represent the true underlying pattern (systematic, consistent error regardless of which training set you used). High bias → underfitting.
- **Variance** — error from the model being too sensitive to the specific training set it saw (it would produce a very different model if trained on a different sample from the same distribution). High variance → overfitting.
- **Irreducible noise** — inherent randomness in the data/labels that no model can ever eliminate.

**The trade-off:** reducing bias (more capacity: bigger network, more features, less regularization) tends to increase variance, and vice versa. Most of what lessons 01-05 taught — architecture size, regularization strength, training duration — are ultimately levers on this one trade-off. There isn't a single "best" setting; it depends on where your current model sits on the curve.

## 4. Hyperparameter Tuning

Hyperparameters (learning rate, hidden layer size, regularization strength, batch size, dropout rate, etc.) aren't learned by gradient descent — you have to search for good values yourself, using the *validation* set as your guide (never the test set).

- **Grid search** — try every combination of a predefined set of values for each hyperparameter. Exhaustive but expensive; grows exponentially with the number of hyperparameters.
- **Random search** — sample random combinations instead of every combination. Often more efficient than grid search in high dimensions, because it doesn't waste evaluations on unimportant hyperparameter axes.
- **Bayesian optimization / more advanced methods** — use the results of previous trials to intelligently choose the next combination to try (not covered in code here, but worth knowing exists — tools like Optuna, Ray Tune implement this).
- **Cross-validation** — instead of one fixed validation split, rotate through K different train/validation splits ("K-fold") and average the results, giving a more robust estimate, especially useful when you have limited data.

## 5. Evaluation Metrics — Why Accuracy Isn't Enough

Accuracy (`correct predictions / total predictions`) can be dangerously misleading on **imbalanced** datasets. Example: if 95% of emails are not spam, a model that predicts "not spam" for *everything* scores 95% accuracy while being completely useless. This is exactly why the following metrics exist.

## 6. Confusion Matrix

The foundation every classification metric below is built from — a table of actual vs. predicted class counts:

|                    | Predicted Positive | Predicted Negative |
|--------------------|:-------------------:|:-------------------:|
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

- **TP** — correctly predicted positive
- **TN** — correctly predicted negative
- **FP** — predicted positive, actually negative ("false alarm" / Type I error)
- **FN** — predicted negative, actually positive ("missed detection" / Type II error)

## 7. Precision, Recall, F1

```
Precision = TP / (TP + FP)     "Of everything I predicted positive, how much was actually positive?"
Recall    = TP / (TP + FN)     "Of everything actually positive, how much did I correctly find?"
F1        = 2 · (Precision · Recall) / (Precision + Recall)     [harmonic mean of the two]
```

- **High precision matters** when false positives are costly (e.g. flagging a legitimate transaction as fraud and blocking a customer's card).
- **High recall matters** when false negatives are costly (e.g. missing an actual cancer diagnosis).
- **F1** balances both into a single number, useful when you need one metric but care about both — though it hides *which* of precision/recall is driving the score, so it's best reported alongside the individual numbers, not instead of them.

## 8. ROC-AUC

A classifier doesn't just output a class — it outputs a *probability*, and you choose a threshold (commonly 0.5) to convert that into a class prediction. The **ROC curve** (Receiver Operating Characteristic) plots the True Positive Rate (Recall) against the False Positive Rate (`FP/(FP+TN)`) as you sweep that threshold from 0 to 1.

**AUC** (Area Under the Curve) summarizes the entire ROC curve into one number:
- `AUC = 1.0` — perfect classifier at every threshold
- `AUC = 0.5` — no better than random guessing
- `AUC < 0.5` — worse than random (though this usually means your labels or predictions are flipped)

**Why it's useful:** unlike accuracy/precision/recall/F1 (all computed at *one* fixed threshold), AUC evaluates the model's *ranking ability* across every possible threshold at once — useful when you don't yet know the right operating threshold for your specific use case, or want a threshold-independent way to compare two models.

## Run the code
[`06-practical-deep-learning.ipynb`]— trains models at three different capacities to directly visualize underfitting/good-fit/overfitting, runs a small grid search over learning rate and hidden size, then computes and visualizes a confusion matrix, precision/recall/F1, and an ROC-AUC curve — first by hand in NumPy, then with `scikit-learn`'s metrics and a PyTorch-trained model.

## You've completed the course
This lesson closes the loop: 01-02 taught you to build and train a network, 03-05 taught you to make that training actually work well, and this lesson teaches you how to know — rigorously, not just by eyeballing loss curves — whether the result is any good.
