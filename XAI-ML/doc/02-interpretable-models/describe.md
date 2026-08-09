# 02 · Interpretable ("Glass-Box") Models

Before reaching for post-hoc explainability tools, always ask: **do I need a black box at all?**

Some models are interpretable *by design* — you can read off exactly why they made a decision, no extra tooling required. This lesson covers the two classic glass-box models and shows their accuracy/interpretability trade-off against a black box (covered in the next lesson).

## What you'll learn
- **Decision Trees** — how splits map directly to human-readable rules, and how `feature_importances_` is derived from impurity reduction.
- **Logistic Regression** — how coefficients directly quantify each feature's linear contribution to the log-odds of the outcome.

## Why this matters
From `01-foundations`: interpretability tends to fall as model complexity rises. Glass-box models sit at the "high interpretability" end of that curve. Whenever a glass-box model reaches acceptable accuracy for your use case, prefer it — it needs no separate explanation layer, has no approximation error, and is trivially auditable.

## Dataset
All notebooks in this repo use the **Breast Cancer Wisconsin** dataset (`sklearn.datasets.load_breast_cancer`) — a binary classification task (Malignant vs Benign) with 30 numeric tumor features. Using one dataset throughout lets you compare explanation methods apples-to-apples.

## Run it
Open [`02-interpretable-models.ipynb`](../../code/02-interpretable-models.ipynb).

## Next
Once you've seen how far a transparent model gets you, move to [`03-shap`](../03-shap) to see how we explain a model that *isn't* transparent.
