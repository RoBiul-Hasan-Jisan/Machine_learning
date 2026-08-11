# Random Search

`RandomizedSearchCV` samples a fixed number of random hyperparameter combinations instead of trying every single one — far more efficient for large or continuous search spaces.

## How it works

```python
from sklearn.model_selection import RandomizedSearchCV
```

Instead of a `param_grid` of discrete lists, you give it `param_distributions` — either lists (sampled uniformly) or actual probability distributions (`scipy.stats.randint`, `scipy.stats.uniform`) for continuous hyperparameters. You set `n_iter` — the fixed number of random combinations to try, regardless of how large the full space is.

```python
from scipy.stats import randint, uniform

param_distributions = {
    "n_estimators": randint(100, 1000),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 20),
    "max_features": uniform(0.3, 0.7)
}
# n_iter=30 means only 30 combinations are tried, no matter how big this space is
```

See `random_search_example.py` for a full runnable comparison against Grid Search on the same problem.

## Why it often beats Grid Search for the same budget
Research (Bergstra & Bengio, 2012) found that in most real hyperparameter spaces, only a few hyperparameters actually matter much. A grid wastes many trials varying an unimportant parameter across all its values while barely varying the important one. Random sampling, by contrast, explores **more distinct values of every parameter** for the same total budget — so it tends to stumble onto good regions of the important parameters faster.

| | Grid Search | Random Search |
|---|---|---|
| Search space | Small, discrete | Large, continuous, or high-dimensional |
| Cost control | Grows multiplicatively with each parameter | Fixed — you set `n_iter` directly |
| Guarantee | Best combination *within the grid* | Good approximation, not guaranteed optimal |
| Typical role | Fine-tuning around a known good region | Initial broad exploration |

## Try it yourself
Run `random_search_example.py` and compare its best CV score and runtime against `06_Grid_Search/grid_search_example.py` — with a fraction of the fits, Random Search often lands very close to Grid Search's result.
