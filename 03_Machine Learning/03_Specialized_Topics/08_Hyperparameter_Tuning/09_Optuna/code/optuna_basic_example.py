

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Quiet down Optuna's default logging so the trial results below are easy to read
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def objective(trial):
    """OBJECTIVE FUNCTION — Optuna calls this once per TRIAL, suggesting new hyperparameters each time."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    return score


# STUDY — the overall optimization session
study = optuna.create_study(direction="maximize", study_name="rf_tuning_demo")

n_trials = 25
study.optimize(objective, n_trials=n_trials)

print(f"Ran {n_trials} trials.")
print(f"\nBest hyperparameters found: {study.best_params}")
print(f"Best CV ROC-AUC: {study.best_value:.4f}")

# Show the trials as a table -- great for spotting which hyperparameter mattered
trials_df = study.trials_dataframe()
print("\nTop 5 trials by score:")
print(trials_df.sort_values("value", ascending=False)
      [["number", "value", "params_n_estimators", "params_max_depth",
        "params_min_samples_split", "params_max_features"]]
      .head(5).to_string(index=False))

# Fit the final model on the full training set with the best found hyperparameters
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print(f"\nFinal test accuracy (evaluated once): {test_score:.4f}")

print("\n--- Parameter importance (which hyperparameter mattered most?) ---")
try:
    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"  {param:22s} {imp:.4f}")
except Exception as e:
    print(f"  (requires additional dependency for importance calc: {e})")
