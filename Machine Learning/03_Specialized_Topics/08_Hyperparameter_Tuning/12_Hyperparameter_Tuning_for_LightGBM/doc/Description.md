# Hyperparameter Tuning for LightGBM

LightGBM shares much with XGBoost but grows trees **leaf-wise** rather than level-wise, which changes which hyperparameter matters most and introduces one XGBoost doesn't have: `num_leaves`.

## The key hyperparameters

| Hyperparameter | Controls | Typical range | Notes |
|---|---|---|---|
| `num_leaves` | Max number of leaves in one tree | 15–255 | **The most important complexity control in LightGBM** — because trees grow leaf-wise, this (not `max_depth`) is the primary lever |
| `max_depth` | Max tree depth | 3–12 (or -1 for unlimited) | Secondary to `num_leaves` here; mostly used as a safety cap to prevent runaway leaf-wise growth on small datasets |
| `learning_rate` | Shrinkage applied to each tree | 0.01–0.3 | Same role as in XGBoost — pairs with `n_estimators`/early stopping |

### Why `num_leaves` matters more here than `max_depth`
XGBoost's default tree growth is **level-wise** — it expands every leaf at the current depth before going deeper, so `max_depth` directly bounds total complexity. LightGBM's default is **leaf-wise** — it always splits whichever leaf reduces loss the most, regardless of depth, which is more efficient but can produce a few very deep, unbalanced branches. `num_leaves` becomes the real complexity dial; `max_depth` (if set) is more of a safety rail.

**Rule of thumb relationship:** for a roughly balanced tree, `num_leaves` ≈ `2^max_depth`. If you set `num_leaves` too high relative to your data size without a `max_depth` cap, LightGBM can overfit quickly — this is the single most common LightGBM tuning mistake.

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,      # high ceiling, use early stopping (folder 10)
    num_leaves=31,          # LightGBM's default -- a reasonable starting point
    max_depth=-1,           # unlimited, letting num_leaves be the main control
    learning_rate=0.05,
)
```

## Full tuning example

`lightgbm_tuning_example.py` tunes `num_leaves`, `max_depth`, and `learning_rate` together with Optuna + early stopping, and separately shows what happens if `num_leaves` is left too high without a `max_depth` cap — a deliberate overfitting demonstration.

```python
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }
    model = lgb.LGBMClassifier(n_estimators=1000, verbosity=-1, **params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    return model.best_score_["valid_0"]["auc"]
```

## Practical tuning order
1. `num_leaves` — the primary complexity control; tune this first and most carefully.
2. `max_depth` — set as a safety cap, especially on smaller datasets, to prevent `num_leaves` from producing runaway-deep branches.
3. `learning_rate` paired with early stopping (not a direct `n_estimators` search).
4. Regularization (`min_child_samples`, `feature_fraction`, `bagging_fraction`) — a further pass once the above are in a good range.

## Try it yourself
Run `lightgbm_tuning_example.py` and compare the tuned model's validation AUC against a deliberately overfit version (`num_leaves=255, max_depth=-1`, no regularization) — the gap between train and validation AUC should be visibly larger for the overfit version.
