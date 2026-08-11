

import warnings
warnings.filterwarnings("ignore")

import optuna
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)

# Test set carved off FIRST, touched only at the very end
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


print("STEP 1 — Baseline with default hyperparameters")

baseline = RandomForestClassifier(random_state=42)
baseline_scores = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"Baseline CV ROC-AUC: {baseline_scores.mean():.4f} +/- {baseline_scores.std():.4f}")


print("STEP 2 — Broad Random Search (wide ranges, cheap exploration)")

wide_param_distributions = {
    "n_estimators": randint(50, 800),
    "max_depth": randint(2, 40),
    "min_samples_split": randint(2, 30),
    "max_features": uniform(0.1, 0.9),
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=wide_param_distributions,
    n_iter=25, cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
print(f"Random Search best CV ROC-AUC: {random_search.best_score_:.4f}")
print(f"Random Search best params: {random_search.best_params_}")


print("STEP 3 — Narrowed Optuna study (informed by Random Search's best region)")


best = random_search.best_params_
# Narrow the search space around what Random Search found, with some margin
depth_lo, depth_hi = max(2, best["max_depth"] - 8), best["max_depth"] + 8
split_lo, split_hi = max(2, best["min_samples_split"] - 5), best["min_samples_split"] + 5
n_est_lo, n_est_hi = max(50, best["n_estimators"] - 150), best["n_estimators"] + 150

print(f"Narrowed ranges — max_depth: [{depth_lo}, {depth_hi}], "
      f"min_samples_split: [{split_lo}, {split_hi}], n_estimators: [{n_est_lo}, {n_est_hi}]")


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", n_est_lo, n_est_hi),
        "max_depth": trial.suggest_int("max_depth", depth_lo, depth_hi),
        "min_samples_split": trial.suggest_int("min_samples_split", split_lo, split_hi),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }
    model = RandomForestClassifier(random_state=42, **params)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(f"\nOptuna best CV ROC-AUC: {study.best_value:.4f}")
print(f"Optuna best params: {study.best_params}")


print("STEP 4 & 5 — Final model, ONE evaluation on the held-out test set")

final_model = RandomForestClassifier(random_state=42, **study.best_params)
final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)
print(f"Final test accuracy (touched exactly once): {test_score:.4f}")


print("SUMMARY — progression across stages")

print(f"{'Stage':30s} {'CV ROC-AUC':>12s}")
print(f"{'1. Baseline (defaults)':30s} {baseline_scores.mean():12.4f}")
print(f"{'2. Random Search (broad)':30s} {random_search.best_score_:12.4f}")
print(f"{'3. Optuna (narrowed)':30s} {study.best_value:12.4f}")
print("\nExpected pattern: steady improvement with diminishing returns at each stage.")
print("If Optuna's narrowed search does WORSE than Random Search's broad one, that's a sign")
print("the narrowing was too aggressive and cut off a better region -- widen it and re-run.")
