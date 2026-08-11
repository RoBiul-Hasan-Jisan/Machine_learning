# Optuna

Optuna is the most widely used modern hyperparameter optimization library. It uses **Tree-structured Parzen Estimator (TPE)** sampling by default — a Bayesian-style approach related to what folder 08 illustrated from scratch, but far more robust and scalable to many hyperparameters.

## Core concepts

### Study
The overall optimization "session" — a collection of trials working toward optimizing one objective.
```python
import optuna
study = optuna.create_study(direction="maximize")  # or "minimize" for loss-like metrics
```

### Trial
A single evaluation of one hyperparameter combination. Inside the objective function, `trial` is used to *suggest* values for each hyperparameter.
```python
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 800)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    ...
```

### Objective Function
The function Optuna repeatedly calls, each time with a new `trial` — it must build a model with the suggested hyperparameters, evaluate it (typically via cross-validation), and return a single score for Optuna to optimize.
```python
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(study.best_params)
print(study.best_value)
```

### Pruning
Stops unpromising trials early — instead of always training a full model, Optuna can check intermediate progress (e.g., after each boosting round) and abandon a trial if it's clearly worse than others so far. This saves enormous amounts of compute on expensive models like gradient boosting.

```python
import optuna
from optuna.integration import XGBoostPruningCallback  # example integration
# See optuna_example.py for a runnable pruning-enabled example with a manual approach
```

## Runnable examples in this folder

- `optuna_basic_example.py` — Study, Trial, and Objective Function on a Random Forest.
- `optuna_pruning_example.py` — demonstrates pruning with a Gradient Boosting model, stopping weak trials early based on intermediate validation performance.

## Optuna vs Grid/Random Search vs from-scratch Bayesian Optimization

| | Grid Search | Random Search | Optuna |
|---|---|---|---|
| Learns from past trials | No | No | Yes (TPE sampler) |
| Handles many hyperparameters well | Poorly (combinatorial blowup) | OK | Well |
| Supports pruning | No | No | Yes |
| Visualization tools | No | No | Yes (`optuna.visualization`) |
| Typical use | Small final fine-tuning | Broad initial search | Default choice for serious tuning |

## Try it yourself
Run `optuna_basic_example.py`, then check `study.trials_dataframe()` to see every trial's parameters and score as a table — this is often the fastest way to spot which hyperparameter actually mattered.
