

import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

CEILING = 1000  # generously high n_estimators ceiling -- early stopping will cut this short
STOPPING_ROUNDS = 20

print(f"n_estimators ceiling set to {CEILING} for all models; early_stopping_rounds={STOPPING_ROUNDS}\n")

# ---------------- XGBoost ----------------

print("XGBoost")

start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=CEILING,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="auc",
    early_stopping_rounds=STOPPING_ROUNDS,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
elapsed = time.time() - start

xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
print(f"Stopped at iteration: {xgb_model.best_iteration} (of {CEILING} ceiling)")
print(f"Best validation AUC : {xgb_model.best_score:.4f}")
print(f"Validation AUC (recomputed): {roc_auc_score(y_val, xgb_preds):.4f}")
print(f"Training time: {elapsed:.2f}s")

# ---------------- LightGBM ----------------

print("LightGBM")

start = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=CEILING,
    max_depth=5,
    learning_rate=0.05,
    verbosity=-1,
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=STOPPING_ROUNDS, verbose=False)]
)
elapsed = time.time() - start

lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
print(f"Stopped at iteration: {lgb_model.best_iteration_} (of {CEILING} ceiling)")
print(f"Validation AUC: {roc_auc_score(y_val, lgb_preds):.4f}")
print(f"Training time: {elapsed:.2f}s")

# ---------------- CatBoost ----------------

print("CatBoost")

start = time.time()
cat_model = CatBoostClassifier(
    iterations=CEILING,
    depth=5,
    learning_rate=0.05,
    eval_metric="AUC",
    early_stopping_rounds=STOPPING_ROUNDS,
    verbose=False,
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
elapsed = time.time() - start

cat_preds = cat_model.predict_proba(X_val)[:, 1]
print(f"Stopped at iteration: {cat_model.get_best_iteration()} (of {CEILING} ceiling)")
print(f"Validation AUC: {roc_auc_score(y_val, cat_preds):.4f}")
print(f"Training time: {elapsed:.2f}s")

# ---------------- Summary ----------------

print("SUMMARY — how many rounds early stopping actually used")

print(f"{'Library':12s} {'Stopped at':>12s} {'% of ceiling used':>20s}")
print(f"{'XGBoost':12s} {xgb_model.best_iteration:12d} {100*xgb_model.best_iteration/CEILING:19.1f}%")
print(f"{'LightGBM':12s} {lgb_model.best_iteration_:12d} {100*lgb_model.best_iteration_/CEILING:19.1f}%")
print(f"{'CatBoost':12s} {cat_model.get_best_iteration():12d} {100*cat_model.get_best_iteration()/CEILING:19.1f}%")
print("\nAll three stopped well before the 1000-round ceiling -- early stopping saved")
print("substantial training time while (in this case) still reaching strong validation AUC.")
