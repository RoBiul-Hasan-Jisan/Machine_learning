#  Stratified K-Fold

## For classification datasets

Plain `KFold` splits data into folds without considering class labels at all — which can accidentally create folds with very different class balances than the overall dataset. On an imbalanced dataset, a fold might end up with very few (or even zero) minority-class examples, making that fold's validation score meaningless and destabilizing your whole hyperparameter comparison.

**`StratifiedKFold` preserves the overall class distribution in every single fold.**

```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Build a moderately imbalanced dataset to make the difference visible
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
print("Overall positive rate:", y.mean().round(3))

model = RandomForestClassifier(n_estimators=200, random_state=42)

print("\n--- Plain KFold: class balance per fold ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (_, val_idx) in enumerate(kf.split(X, y)):
    print(f"  Fold {i+1}: positive rate = {y[val_idx].mean():.3f}")

print("\n--- StratifiedKFold: class balance per fold ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (_, val_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {i+1}: positive rate = {y[val_idx].mean():.3f}")

# Compare final CV scores
kf_scores = cross_val_score(model, X, y, cv=kf, scoring="roc_auc")
skf_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
print(f"\nKFold ROC-AUC          : {kf_scores.mean():.4f} +/- {kf_scores.std():.4f}")
print(f"StratifiedKFold ROC-AUC: {skf_scores.mean():.4f} +/- {skf_scores.std():.4f}")
```

### Rule of thumb
**Always use `StratifiedKFold` (not plain `KFold`) for classification tasks** — pass it directly as the `cv` argument to `GridSearchCV`, `RandomizedSearchCV`, or `cross_val_score`. It costs nothing extra and removes a real source of noisy, misleading hyperparameter comparisons.

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=model,
    param_grid={"max_depth": [3, 5, 10]},
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # <- use this, not a plain int
    scoring="roc_auc"
)
```

> Passing `cv=5` (an integer) to `GridSearchCV`/`RandomizedSearchCV` for a classifier actually already defaults to `StratifiedKFold` internally in scikit-learn — but being explicit is safer and makes your intent clear, especially once `shuffle=True` and a `random_state` matter for reproducibility.

## Try it yourself
Rebuild the imbalance even more severe (`weights=[0.97, 0.03]`) and re-run both KFold and StratifiedKFold — watch a plain KFold fold occasionally drop to a positive rate of 0.00 or 0.01, and see how that destabilizes the mean/std of the CV scores.
