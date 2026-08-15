#  Why Hyperparameter Tuning

## Parameters vs Hyperparameters

| | Parameters | Hyperparameters |
|---|---|---|
| **What** | Values the model *learns* from data during training | Values *you set* before training begins |
| **Examples** | Linear regression coefficients, neural net weights, tree split points | `learning_rate`, `max_depth`, `n_estimators`, `C`, `k` in KNN |
| **Who sets them** | The optimization algorithm (gradient descent, tree-building) | The practitioner (you) |
| **Where they live** | Inside the fitted model object (`model.coef_`, `model.feature_importances_`) | Passed into the model constructor (`RandomForestClassifier(n_estimators=200)`) |

A simple mental model: **parameters are what the model figures out; hyperparameters are what you tell it before it starts figuring things out.** Hyperparameter tuning is the process of searching for the hyperparameter values that make the *learning process itself* produce the best possible model.

```python
from sklearn.ensemble import RandomForestClassifier

# n_estimators, max_depth, min_samples_split = HYPERPARAMETERS (you choose these)
model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5)
model.fit(X_train, y_train)

# feature_importances_, the actual tree splits = PARAMETERS (learned from data)
print(model.feature_importances_)
```

## Underfitting vs Overfitting

Hyperparameters directly control where a model lands on the underfitting ↔ overfitting spectrum.

| | Underfitting | Overfitting |
|---|---|---|
| **Symptom** | Poor performance on both train and validation data | Great performance on train, much worse on validation |
| **Cause** | Model too simple / too constrained for the data's patterns | Model too complex / too free, memorizes noise |
| **Hyperparameter direction** | Increase complexity (deeper trees, more estimators, smaller regularization) | Decrease complexity (shallower trees, fewer estimators, more regularization, early stopping) |

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Underfit-prone: too shallow to capture real patterns
underfit = DecisionTreeClassifier(max_depth=1)

# Overfit-prone: unlimited depth memorizes training data
overfit = DecisionTreeClassifier(max_depth=None)

# Somewhere in between is usually where the best hyperparameters live
balanced = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
```

### Why this justifies the whole chapter
Every hyperparameter is essentially a **complexity dial**. Tuning is the systematic process of finding the dial settings that sit at the sweet spot of the bias-variance tradeoff — enough complexity to capture real patterns, not so much that the model starts fitting noise. The rest of this chapter is about *how* to search that space efficiently and *honestly* (without accidentally cheating by peeking at test data).

## Try it yourself
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for depth in [1, 2, 4, 8, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"max_depth={str(depth):5s} | train acc={train_acc:.3f} | val acc={val_acc:.3f} | gap={train_acc-val_acc:.3f}")
```
Run this and watch the gap between train and validation accuracy grow as `max_depth` increases — that gap *is* overfitting, made visible.
