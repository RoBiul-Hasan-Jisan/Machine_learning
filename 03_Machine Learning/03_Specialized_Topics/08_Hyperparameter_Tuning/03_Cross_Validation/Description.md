#  Cross-Validation

## Why CV is needed

A single train/validation split gives you **one** performance number per hyperparameter combination — and that number depends partly on which rows randomly landed in validation. A slightly different split could crown a different "best" hyperparameter combination. Cross-validation fixes this by evaluating each combination across **multiple** splits and averaging, giving a far more stable and trustworthy comparison.

This matters enormously for tuning specifically: you're comparing many candidate hyperparameter settings against each other, and you want that comparison to reflect real generalization, not the luck of one split.

## CV workflow

1. Split the training data into K folds.
2. For each hyperparameter combination being tested:
   - Train K models, each time holding out one fold as validation and training on the rest.
   - Average the K validation scores → this is that combination's CV score.
3. Compare CV scores across all combinations, pick the best one.
4. Retrain a final model on **all** training data using the winning hyperparameters.
5. Evaluate that final model **once** on the untouched test set.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

# cv=5 -> 5-Fold cross-validation, returns one score per fold
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print("Fold scores:", scores.round(4))
print(f"Mean CV score: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### Why report the standard deviation, not just the mean
Two hyperparameter combinations with the same mean CV score aren't necessarily equally good — the one with **lower variance across folds** is more reliably going to perform that way on new data. Always look at both numbers, not just the mean, when comparing candidates.

## Try it yourself
Run `cross_val_score` for 3 different `max_depth` values on a Random Forest, and for each print both the mean and std of the fold scores. Notice that the best mean doesn't always come with the tightest std — that tension is part of real-world model selection.
