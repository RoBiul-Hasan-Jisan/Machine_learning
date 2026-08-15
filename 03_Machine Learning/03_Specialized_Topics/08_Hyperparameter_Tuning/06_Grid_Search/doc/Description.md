#  Grid Search

`GridSearchCV` exhaustively tries **every combination** of hyperparameter values you specify, scoring each with cross-validation.

## How it works

```python
from sklearn.model_selection import GridSearchCV
```

You give it a `param_grid` — a dictionary mapping hyperparameter names to lists of values to try. It builds the full Cartesian product of all combinations, trains + cross-validates a model for each, and reports the best one.

```python
param_grid = {
    "n_estimators": [100, 300, 500],   # 3 values
    "max_depth": [5, 10, None],        # 3 values
    "min_samples_split": [2, 5]        # 2 values
}
# Total combinations = 3 x 3 x 2 = 18
# With cv=5, that's 18 x 5 = 90 model fits
```

See `grid_search_example.py` in this folder for a full runnable example — it fits a `RandomForestClassifier` on the Breast Cancer dataset, reports the best parameters, and shows the total number of fits performed.

## Pros and cons

| Pros | Cons |
|---|---|
| Guaranteed to find the best combination **within the grid you defined** | Cost grows multiplicatively — adding one more hyperparameter or value can blow up runtime |
| Simple, deterministic, easy to reason about | Wastes compute on combinations in clearly bad regions |
| Great for small, well-understood search spaces | Struggles with continuous hyperparameters (you have to discretize them yourself) |

## When to use it
Grid Search shines when you already have a rough idea of the good region (e.g., from a prior Random Search pass — see folder 07) and want to exhaustively fine-tune around it with just 2–3 hyperparameters and a handful of values each.

## Try it yourself
Run `grid_search_example.py`, then try adding a fourth hyperparameter with 3 more values and watch how many total fits are needed — that's the "curse of dimensionality" of grid search in action.
