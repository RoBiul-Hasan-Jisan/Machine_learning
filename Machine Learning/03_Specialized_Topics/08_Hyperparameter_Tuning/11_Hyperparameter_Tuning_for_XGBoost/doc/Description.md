#  Hyperparameter Tuning for XGBoost

The handful of hyperparameters that matter most for XGBoost, what each one controls, and how to tune them together.

## The key hyperparameters

| Hyperparameter | Controls | Typical range | Effect of increasing it |
|---|---|---|---|
| `n_estimators` | Number of boosting rounds (trees) | 100–2000 (use early stopping instead of tuning this directly) | More capacity, risk of overfitting if unchecked |
| `max_depth` | Max depth of each individual tree | 3–10 | Deeper trees = more complex splits per tree = higher overfitting risk |
| `learning_rate` (`eta`) | How much each tree's prediction shrinks before being added to the ensemble | 0.01–0.3 | Lower = needs more trees but generalizes better; classic tradeoff with `n_estimators` |
| `subsample` | Fraction of training rows sampled (without replacement) for each tree | 0.5–1.0 | Lower = more randomness/regularization, less overfitting, but noisier trees |
| `colsample_bytree` | Fraction of features sampled for each tree | 0.5–1.0 | Lower = more randomness/regularization, forces the model not to over-rely on a few features |

### The `learning_rate` / `n_estimators` relationship
These two are tightly linked: a **lower learning rate needs more trees** to reach the same overall fit. The standard practice — set a low learning rate (e.g., 0.01–0.05) with a high `n_estimators` ceiling, and let **early stopping** (folder 10) find the right number of rounds automatically, rather than tuning `n_estimators` directly.

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=2000,       # high ceiling, early stopping will cut it short
    learning_rate=0.03,      # low, for better generalization
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=50,
)
```

## Full tuning example

`xgboost_tuning_example.py` in this folder combines Optuna (folder 09) with early stopping (folder 10) to tune all five key hyperparameters together — the practical, realistic workflow rather than tuning each in isolation.

```python
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = xgb.XGBClassifier(
        n_estimators=1000, eval_metric="auc", early_stopping_rounds=30, **params
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.best_score
```

## Practical tuning order (most to least impactful, generally)
1. `max_depth` and `learning_rate` together — these have the strongest effect on the bias-variance tradeoff.
2. `subsample` and `colsample_bytree` — regularization knobs, tune after the above are roughly right.
3. `n_estimators` — let early stopping handle this automatically rather than searching it directly.
4. Finer regularization (`reg_alpha`, `reg_lambda`, `min_child_weight`) — worth a pass once the above are dialed in, for squeezing out the last bit of performance.

