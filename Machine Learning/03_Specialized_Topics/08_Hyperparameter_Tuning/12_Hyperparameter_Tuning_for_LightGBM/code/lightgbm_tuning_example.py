

import warnings
warnings.filterwarnings("ignore")
import optuna
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val2, X_test, y_val2, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)


def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }
    model = lgb.LGBMClassifier(n_estimators=1000, verbosity=-1, random_state=42, **params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val2, y_val2)], eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    trial.set_user_attr("best_iteration", model.best_iteration_)
    preds = model.predict_proba(X_val2)[:, 1]
    return roc_auc_score(y_val2, preds)


study = optuna.create_study(direction="maximize")
n_trials = 25
study.optimize(objective, n_trials=n_trials)

print(f"Ran {n_trials} Optuna trials.")
print(f"\nBest hyperparameters: {study.best_params}")
print(f"Best validation AUC: {study.best_value:.4f}")
print(f"Boosting rounds used (via early stopping): {study.best_trial.user_attrs['best_iteration']}")

# Train final tuned model and evaluate on the held-out test set
tuned_model = lgb.LGBMClassifier(n_estimators=1000, verbosity=-1, random_state=42, **study.best_params)
tuned_model.fit(
    X_train, y_train,
    eval_set=[(X_val2, y_val2)], eval_metric="auc",
    callbacks=[lgb.early_stopping(30, verbose=False)]
)
tuned_test_auc = roc_auc_score(y_test, tuned_model.predict_proba(X_test)[:, 1])
tuned_train_auc = roc_auc_score(y_train, tuned_model.predict_proba(X_train)[:, 1])
print(f"\nTuned model — train AUC: {tuned_train_auc:.4f} | test AUC: {tuned_test_auc:.4f} "
      f"| gap: {tuned_train_auc - tuned_test_auc:.4f}")

# ---- Deliberate overfitting demo: num_leaves too high, no max_depth cap ----
# Uses a noisier, smaller synthetic dataset so the overfitting effect is clearly visible
# (breast cancer is too easy/separable to show a strong gap)

print("OVERFITTING DEMO — num_leaves=255, max_depth=-1 (unlimited), no regularization")


from sklearn.datasets import make_classification
Xn, yn = make_classification(
    n_samples=400, n_features=20, n_informative=6, n_redundant=4,
    flip_y=0.08, random_state=42
)
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.3, stratify=yn, random_state=42)

tuned_small = lgb.LGBMClassifier(
    n_estimators=200, num_leaves=15, max_depth=4, learning_rate=0.05,
    min_child_samples=20, verbosity=-1, random_state=42
)
tuned_small.fit(Xn_train, yn_train)
tuned_train_auc = roc_auc_score(yn_train, tuned_small.predict_proba(Xn_train)[:, 1])
tuned_test_auc = roc_auc_score(yn_test, tuned_small.predict_proba(Xn_test)[:, 1])
print(f"Regularized model  — train AUC: {tuned_train_auc:.4f} | test AUC: {tuned_test_auc:.4f} "
      f"| gap: {tuned_train_auc - tuned_test_auc:.4f}")

overfit_model = lgb.LGBMClassifier(
    n_estimators=500, num_leaves=255, max_depth=-1,
    learning_rate=0.2, min_child_samples=1, verbosity=-1, random_state=42
)
overfit_model.fit(Xn_train, yn_train)

overfit_train_auc = roc_auc_score(yn_train, overfit_model.predict_proba(Xn_train)[:, 1])
overfit_test_auc = roc_auc_score(yn_test, overfit_model.predict_proba(Xn_test)[:, 1])
print(f"Overfit model      — train AUC: {overfit_train_auc:.4f} | test AUC: {overfit_test_auc:.4f} "
      f"| gap: {overfit_train_auc - overfit_test_auc:.4f}")

print(f"\nRegularized model gap: {tuned_train_auc - tuned_test_auc:.4f}")
print(f"Overfit model gap:     {overfit_train_auc - overfit_test_auc:.4f}")
print("\nA visibly larger train-test gap for the 'overfit' config confirms num_leaves")
print("really is the primary complexity/overfitting lever in LightGBM.")
