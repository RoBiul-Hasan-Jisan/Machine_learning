#  Early Stopping

Gradient boosting libraries build models iteratively — adding one tree at a time. Early stopping monitors validation performance during this process and **halts training automatically** once more trees stop helping (or start hurting via overfitting), instead of training a fixed number of rounds decided in advance.

This is a hyperparameter tuning technique in its own right: it effectively tunes `n_estimators` *automatically* and *for free* as part of a single training run, rather than needing to search over it explicitly.

## XGBoost

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=1000,          # set a high ceiling -- early stopping will cut it short
    max_depth=5,
    learning_rate=0.05,
    eval_metric="auc",
    early_stopping_rounds=20,   # stop if no improvement for 20 consecutive rounds
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print("Best iteration:", model.best_iteration)
print("Best validation AUC:", model.best_score)
```

## LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)
print("Best iteration:", model.best_iteration_)
```

## CatBoost

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    depth=5,
    learning_rate=0.05,
    eval_metric="AUC",
    early_stopping_rounds=20,
    verbose=False
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
print("Best iteration:", model.get_best_iteration())
```

See `early_stopping_example.py` for a full runnable comparison of all three libraries side by side, including how many rounds each actually used before stopping.

## Why this matters for tuning
Without early stopping, `n_estimators` becomes just another hyperparameter you'd need to search over (as in folders 06/07/09) — and getting it wrong in either direction costs you: too few rounds underfits, too many overfits and wastes compute. Early stopping removes that search entirely for one of the most impactful hyperparameters in boosting models, letting each training run find its own right length automatically.

**Best practice:** set `n_estimators` (or `iterations`) to a generously high ceiling, always pass a validation set, and let early stopping decide where to actually cut off — then focus your explicit tuning (Grid/Random/Optuna) on the other hyperparameters (`max_depth`, `learning_rate`, `subsample`, etc. — see folders 11–12).

## Try it yourself
Run `early_stopping_example.py` and compare the `best_iteration` each library settles on against the `n_estimators=1000` ceiling you gave it — notice how much unnecessary training was avoided.
