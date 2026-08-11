# Train / Validation / Test Split

Hyperparameter tuning needs **three** distinct data buckets — not two — because tuning itself is a form of "fitting" that can overfit if you're not careful.

## The three sets

| Set | Used for | Touched how often |
|---|---|---|
| **Train** | Fitting model parameters (weights, tree splits) for each hyperparameter combination tried | Many times — once per hyperparameter combination |
| **Validation** | Comparing hyperparameter combinations and picking the winner | Many times — once per combination, to score it |
| **Test** | Final, honest estimate of how the *tuned* model performs on unseen data | **Exactly once**, at the very end |

### Why not just train/test?
If you tune hyperparameters by repeatedly checking performance on your "test" set, you're implicitly *fitting the hyperparameters to that test set* — its role quietly becomes a second validation set, and your final reported number stops being honest. The validation set exists specifically to absorb that repeated probing, keeping the test set clean for one final, trustworthy check.

## Code

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Step 1: carve off the test set first — set it aside and don't touch it again
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: split the remainder into train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)
# Overall: ~60% train / 20% validation / 20% test

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
```

### `stratify=y`
For classification, always stratify — it keeps class proportions consistent across all three sets, which matters even more once you're comparing many hyperparameter combinations against each other (an unlucky imbalanced validation split could crown the wrong "winner").

### In practice: most tuning uses CV instead of a fixed validation set
A single validation split is a reasonable starting point, but in practice most hyperparameter tuning (see folders 03–09) replaces the single validation set with **cross-validation** on the training data — it gives a more stable estimate of which hyperparameters are actually best, while the test set's role (touched once, at the end) stays exactly the same.

```python
# The pattern used throughout the rest of this chapter:
# X_train_full -> used inside cross-validation for tuning
# X_test       -> touched exactly once, after tuning is finished
```

