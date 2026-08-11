

import optuna
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def objective(trial):
    """
    Trains a Gradient Boosting model in stages (warm_start), reporting intermediate
    validation performance to Optuna after each stage. Optuna can then PRUNE
    (abandon) this trial early if it's clearly underperforming other trials
    at the same stage -- without needing to finish training it.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }

    model = GradientBoostingClassifier(
        n_estimators=1, warm_start=True, random_state=42, **params
    )

    n_stages = 20
    stage_size = 10  # add 10 more trees at each stage
    for stage in range(n_stages):
        model.n_estimators += stage_size
        model.fit(X_train, y_train)

        val_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        # Report intermediate value to Optuna at this stage
        trial.report(val_score, step=stage)

        # Ask Optuna: should this trial be pruned (abandoned) based on progress so far?
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_score


# MedianPruner: prunes a trial if its intermediate score is worse than the median
# of other trials' scores at the same step -- a simple, effective default pruning strategy
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
)

n_trials = 30
study.optimize(objective, n_trials=n_trials)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"Total trials: {n_trials}")
print(f"Completed (ran to the end): {len(completed)}")
print(f"Pruned (abandoned early): {len(pruned)}")
print(f"\nBest hyperparameters: {study.best_params}")
print(f"Best validation ROC-AUC: {study.best_value:.4f}")

print("\nWhy this matters: pruned trials never finished all 20 stages of training --")
print("Optuna cut them off early once it was clear they weren't competitive, saving")
print("all the compute that would have gone into finishing them.")

total_stages_if_no_pruning = n_trials * 20
total_stages_actually_run = sum(
    (t.last_step + 1 if t.last_step is not None else 20) for t in study.trials
)
print(f"\nStages that WOULD have run without pruning: {total_stages_if_no_pruning}")
print(f"Stages ACTUALLY run with pruning enabled  : {total_stages_actually_run}")
print(f"Compute saved: {100 * (1 - total_stages_actually_run / total_stages_if_no_pruning):.1f}%")
