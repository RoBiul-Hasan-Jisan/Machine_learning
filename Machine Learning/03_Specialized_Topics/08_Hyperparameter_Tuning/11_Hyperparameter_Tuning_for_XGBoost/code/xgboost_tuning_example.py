
import optuna
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val2, X_test, y_val2, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)
# X_val2 -> used inside tuning for early stopping / scoring
# X_test -> held out completely, touched once at the very end


def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = xgb.XGBClassifier(
        n_estimators=1000,          # high ceiling; early stopping decides the real number
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val2, y_val2)], verbose=False)
    trial.set_user_attr("best_iteration", model.best_iteration)
    return model.best_score


study = optuna.create_study(direction="maximize")
n_trials = 25
study.optimize(objective, n_trials=n_trials)

print(f"Ran {n_trials} Optuna trials (each with early stopping).")
print(f"\nBest hyperparameters: {study.best_params}")
print(f"Best validation AUC: {study.best_value:.4f}")
print(f"Boosting rounds used by the best trial (via early stopping): "
      f"{study.best_trial.user_attrs['best_iteration']}")

# Train the final model with the best hyperparameters found
final_model = xgb.XGBClassifier(
    n_estimators=1000,
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42,
    **study.best_params,
)
final_model.fit(X_train, y_train, eval_set=[(X_val2, y_val2)], verbose=False)

test_preds = final_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_preds)
print(f"\nFinal held-out test AUC (touched once): {test_auc:.4f}")

print("\n--- Parameter importance ---")
try:
    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"  {param:20s} {imp:.4f}")
except Exception as e:
    print(f"  (unable to compute: {e})")
