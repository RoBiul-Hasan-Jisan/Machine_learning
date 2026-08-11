# K-Fold Cross-Validation

The specific mechanics of splitting data into K folds — the engine underneath most hyperparameter tuning tools.

## How K-Fold works

Given K folds, the data is partitioned into K equal-sized chunks. The model is trained K times: each time, one chunk is held out as validation and the rest (K−1 chunks) are used for training. Every row gets used for validation exactly once, and for training K−1 times.

```
Fold 1: [VAL ][ TRAIN ][ TRAIN ][ TRAIN ][ TRAIN ]
Fold 2: [TRAIN][ VAL   ][ TRAIN ][ TRAIN ][ TRAIN ]
Fold 3: [TRAIN][ TRAIN ][ VAL   ][ TRAIN ][ TRAIN ]
Fold 4: [TRAIN][ TRAIN ][ TRAIN ][ VAL   ][ TRAIN ]
Fold 5: [TRAIN][ TRAIN ][ TRAIN ][ TRAIN ][ VAL   ]
```

## 5-Fold vs 10-Fold

| | 5-Fold | 10-Fold |
|---|---|---|
| Training set size per fold | 80% of data | 90% of data |
| Validation set size per fold | 20% of data | 10% of data |
| Number of models trained | 5 | 10 |
| Compute cost | Lower | Higher (2x the fits) |
| Estimate variance | Slightly higher | Slightly lower (more folds → more averaging) |
| Common default | Yes — the most common choice | Used when data is limited and every row of validation signal matters |

**Rule of thumb:** 5-Fold is the standard default for most tuning workflows — it balances compute cost against estimate stability well. Move to 10-Fold when your dataset is small enough that you want to squeeze out a more stable estimate, and can afford double the training runs. LOOCV (K = number of samples) is the extreme end — thorough but usually too expensive for tuning loops that already multiply cost by every hyperparameter combination.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=200, random_state=42)

kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
scores_5fold = cross_val_score(model, X, y, cv=kf5, scoring="roc_auc")
print(f"5-Fold  — mean: {scores_5fold.mean():.4f} | std: {scores_5fold.std():.4f}")

kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
scores_10fold = cross_val_score(model, X, y, cv=kf10, scoring="roc_auc")
print(f"10-Fold — mean: {scores_10fold.mean():.4f} | std: {scores_10fold.std():.4f}")
```

### `shuffle=True` matters
Without shuffling, `KFold` splits the data in its original order — if your data happens to be sorted (by date, by class, by any pattern), some folds could end up wildly unrepresentative. Always set `shuffle=True` (with a `random_state` for reproducibility) unless your data has a specific order you intentionally want to preserve (e.g., time series — see `TimeSeriesSplit` instead in that case).

## Try it yourself
Run both 5-Fold and 10-Fold on the same model/dataset and compare not just the mean scores but the total wall-clock time each took — this tradeoff becomes very real once you multiply it by dozens of hyperparameter combinations in Grid/Random Search.
