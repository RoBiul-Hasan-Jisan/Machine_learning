#  Cross-Validation Pipeline

## Learning Objectives

- Explain why k-fold cross-validation gives a more reliable estimate than a single validation split
- Combine `Pipeline` with `cross_val_score` and `GridSearchCV` so preprocessing is refit correctly inside every fold
- Choose the right CV strategy for classification, regression, time series, and grouped data

## The Problem

A single train/validation split (Lesson 02) gives one number, and that number has variance: a different random split could give a noticeably different score, especially on smaller datasets. Cross-validation averages over several splits, giving both a more stable estimate and a sense of how much that estimate varies.

The catch: every preprocessing step that learns from data (scaling, imputation, encoding, feature selection) must be refit inside each fold using only that fold's training portion. Do this by hand and it's easy to get wrong. Pipeline (Lesson 03) plus scikit-learn's CV utilities do it automatically.

## The Concept

### K-fold cross-validation

```
Data split into k folds. Each fold takes a turn as validation; the rest are training.

Fold 1: [ VAL ][ train ][ train ][ train ][ train ]  -> score_1
Fold 2: [ train ][ VAL ][ train ][ train ][ train ]  -> score_2
Fold 3: [ train ][ train ][ VAL ][ train ][ train ]  -> score_3
Fold 4: [ train ][ train ][ train ][ VAL ][ train ]  -> score_4
Fold 5: [ train ][ train ][ train ][ train ][ VAL ]  -> score_5

Final estimate: mean(score_1..score_5), with std as a measure of variance
```

Every row gets used for both training and validation across the k folds, but never both *at the same time* within a single fold. Typical k is 5 or 10 — higher k means more training data per fold (less bias) but more compute and more correlated folds.

### Why Pipeline + cross_val_score is the safe combination

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
```

When `pipeline` is a `Pipeline` object, `cross_val_score` clones it fresh for each fold and calls `fit` on that fold's training portion only — meaning the scaler, imputer, encoder, and feature selector inside the pipeline are all refit per fold automatically. If you instead scaled the data once outside the loop and then ran manual k-fold splits, you'd leak fold-specific validation statistics into training, inflating your score (this is exactly the leakage pattern from Lessons 02 and 06).

### Choosing a CV strategy

| Strategy | Use for |
|---|---|
| `KFold` | Regression, i.i.d. numeric data |
| `StratifiedKFold` | Classification, especially with class imbalance — preserves class proportions in every fold |
| `TimeSeriesSplit` | Time-ordered data — each fold trains only on the past relative to its validation window |
| `GroupKFold` | Grouped data — ensures no group (customer, patient, ...) appears in both train and validation within a fold |

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
```

`cross_val_score(..., cv=5)` with an integer defaults to `KFold` for regression and `StratifiedKFold` for classification — but being explicit makes the choice visible and lets you control `shuffle` and `random_state`.

### Cross-validation + hyperparameter tuning: GridSearchCV

Tuning hyperparameters using a single validation split risks overfitting to that one split's noise. `GridSearchCV` wraps cross-validation around the search: for each candidate hyperparameter combination, it runs full k-fold CV and averages the score, refitting all pipeline steps inside every fold of every candidate.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "select__k": [5, 10, 15],
    "model__C": [0.01, 0.1, 1, 10],
}

search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True
)
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)          # mean CV score of the best combination
best_pipeline = search.best_estimator_   # refit on the FULL X_train with best params
```

`refit=True` (the default) automatically refits the best-scoring pipeline on the entire `X_train` once the search is done — that final pipeline is what you evaluate on the untouched test set and what you'd persist (Lesson 09).

For large grids, `RandomizedSearchCV` samples a fixed number of random combinations instead of trying every combination — often nearly as good at a fraction of the compute.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    "model__C": loguniform(1e-3, 1e2),
    "select__k": [5, 8, 10, 12, 15],
}

random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=25, cv=5, scoring="roc_auc",
    random_state=42, n_jobs=-1,
)
random_search.fit(X_train, y_train)
```

### Nested cross-validation (when you need an honest tuned score)

If you report `search.best_score_` as your final performance estimate, you're reusing the same folds both to *choose* hyperparameters and to *estimate* performance — a mild form of the same leakage problem, at the level of model selection rather than preprocessing. Nested CV fixes this with an outer loop for evaluation and an inner loop for tuning:

```python
from sklearn.model_selection import cross_val_score

nested_scores = cross_val_score(
    GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc"),
    X_train, y_train, cv=5,  # outer loop
)
```

This is more expensive (5 × 5 = 25 pipeline fits per parameter combination) but gives an unbiased estimate of how the *tuning process itself* generalizes, not just the final chosen model.

See `code/cv_pipeline_demo.py` for a runnable comparison of a single split vs k-fold CV, plus `GridSearchCV` and nested CV on the same pipeline.

## Exercises

1. Run the same pipeline through a single 80/20 split 10 times with different random seeds, then run 5-fold CV once. Compare the spread of the single-split scores to the CV fold scores.
2. Use `StratifiedKFold` vs plain `KFold` on an imbalanced classification dataset and compare the per-fold class proportions.
3. Run `GridSearchCV` and print `search.cv_results_` as a DataFrame, sorted by mean test score, to see every candidate's performance, not just the best one.
4. Implement nested CV for a small param grid and compare the nested score to `GridSearchCV(...).best_score_` on the same data. Explain the direction of any difference.

## Key Terms

| Term | What it actually means |
|---|---|
| K-fold cross-validation | Splitting data into k folds and rotating which fold is used for validation, averaging the resulting scores |
| StratifiedKFold | K-fold splitting that preserves class proportions in every fold |
| GridSearchCV | Exhaustive search over a hyperparameter grid, using cross-validation to score each combination |
| RandomizedSearchCV | Hyperparameter search over a fixed number of randomly sampled combinations, cheaper than a full grid |
| Nested cross-validation | Cross-validation with an outer loop for performance estimation and an inner loop for hyperparameter tuning, avoiding tuning-related optimism |
