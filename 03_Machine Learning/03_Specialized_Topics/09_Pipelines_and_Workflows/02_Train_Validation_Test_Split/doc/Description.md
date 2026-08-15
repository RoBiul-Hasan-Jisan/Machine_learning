# Train / Validation / Test Split

## Learning Objectives

- Explain why you need three data splits, not two, and what each one protects against
- Implement holdout splits correctly for i.i.d., time series, and grouped data
- Identify the most common forms of data leakage and how splitting order prevents them

## The Problem

If you tune your model, then evaluate it, on the same data, your evaluation number is a lie. It measures how well the model (and your tuning decisions) fit that specific dataset, not how well it will perform on new data. You need data the model — and you — never got to see or make decisions about, until the very end.

## The Concept

### Three splits, three jobs

```
Full Dataset
    │
    ├── Train (≈60-80%)       → fit model parameters (weights, tree splits, ...)
    ├── Validation (≈10-20%)  → compare models / tune hyperparameters
    └── Test (≈10-20%)        → final, one-time estimate of real-world performance
```

- **Train**: the model learns its parameters from this data.
- **Validation**: used repeatedly while you try different models and hyperparameters. Because you look at it many times and make decisions based on it, it stops being a clean estimate of generalization — you've indirectly fit to it.
- **Test**: touched exactly once, at the very end, after every modeling decision is frozen. This is your honest estimate of production performance.

A common mistake is using only train/test and tuning hyperparameters against the test set. At that point the test set has become a second validation set, and you have no honest number left. This is why Lesson 08 (cross-validation) replaces a single validation split with k-fold CV — it uses the training data more efficiently while keeping the test set untouched.

### Holdout validation

The basic pattern:

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
# Result: 70% train, 15% val, 15% test
```

`stratify=y` keeps class proportions consistent across splits — essential for imbalanced classification, where a random split could otherwise put almost all the minority class into one split by chance.

### Data leakage: the split order matters

Leakage means information from outside the training set influences the model. The single most common cause is doing a data-dependent operation (scaling, imputation, feature selection) **before** splitting, using statistics computed from the full dataset.

```
WRONG:
  scaler.fit(X_all)          # sees test data statistics
  X_all_scaled = scaler.transform(X_all)
  X_train, X_test = train_test_split(X_all_scaled, ...)

RIGHT:
  X_train, X_test = train_test_split(X_all, ...)
  scaler.fit(X_train)               # only sees training data
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)   # transform only, no fit
```

This is the exact problem `Pipeline` (Lesson 03) and `cross_val_score` solve automatically: they refit every preprocessing step inside each fold, so no fold's transformation is contaminated by the others.

Other common leakage sources:
- **Duplicate or near-duplicate rows** split across train and test (the model "memorizes" a row it technically never trained on).
- **Feature selection or target encoding computed on the full dataset** before splitting.
- **Time leakage**: using future information to predict the past (see below).
- **Group leakage**: the same entity (patient, user, store) appearing in both train and test, when the real prediction task is about generalizing to *new* entities.

### Splitting non-i.i.d. data

Plain `train_test_split` assumes rows are independent and identically distributed. Two common cases where that assumption is wrong:

**Time series** — never split randomly. Train on the past, validate/test on the future, or you leak future information backward.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
```

**Grouped data** — when multiple rows belong to the same entity (e.g. multiple transactions per customer) and the real task is generalizing to unseen entities, split by group so no entity appears in both train and test.

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=customer_id))
```

### Best practices

- Split before any fitting of scalers, imputers, encoders, or feature selectors.
- Use `stratify` for classification, especially with class imbalance.
- Use `TimeSeriesSplit` for anything with a time axis; never shuffle time series.
- Use `GroupShuffleSplit` / `GroupKFold` when rows are grouped and generalization to new groups is the goal.
- Set `random_state` for reproducibility, and document the split ratios and random seed used.
- Touch the test set exactly once, after all modeling decisions are finalized.

See `code/split_demo.py` for a runnable comparison of a naive (leaky) split vs a correct one, and a grouped/time-series example.

## Exercises

1. Take a dataset with class imbalance (e.g. 5% positive class). Compare stratified vs non-stratified `train_test_split` across 20 random seeds — measure the variance in positive-class proportion in the test set for each.
2. Deliberately introduce leakage by scaling before splitting, then compare test accuracy against the correct order. Explain the direction and size of the discrepancy.
3. Simulate grouped data (e.g. 5 rows per customer, 100 customers). Compare a random split vs `GroupShuffleSplit` and show how a random split can put the same customer in both train and test.

## Key Terms

| Term | What it actually means |
|---|---|
| Holdout validation | Setting aside a fixed subset of data (validation or test) that the model does not train on, to estimate generalization |
| Data leakage | Information from outside the training set — including future data, other rows, or full-dataset statistics — reaching the model or its preprocessing |
| Stratified split | A split that preserves the class proportions of the original dataset in each subset |
| Group leakage | The same entity appearing in both train and test when the task requires generalizing to unseen entities |
| Time leakage | Using information from the future to predict the past, usually via a non-chronological split |
