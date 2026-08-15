
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}
total_combinations = 3 * 3 * 2
print(f"Total hyperparameter combinations in the grid: {total_combinations}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)

start = time.time()
grid_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"\nTotal model fits performed: {total_combinations * cv.get_n_splits()}")
print(f"Time taken: {elapsed:.2f} seconds")
print(f"\nBest hyperparameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"\nFinal test accuracy (evaluated once): {test_score:.4f}")

# Show the top 5 combinations by mean CV score, for context
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
top5 = results.sort_values("mean_test_score", ascending=False).head(5)
print("\nTop 5 combinations by mean CV score:")
print(top5[["params", "mean_test_score", "std_test_score"]].to_string(index=False))
