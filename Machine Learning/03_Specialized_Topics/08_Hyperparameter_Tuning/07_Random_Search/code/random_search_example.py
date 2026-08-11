

import time
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# A MUCH larger search space than the Grid Search example — continuous distributions,
# something Grid Search can't handle without manually discretizing
param_distributions = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 20),
    "max_features": uniform(0.3, 0.7),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_iter = 20  # only 20 combinations tried, regardless of the space's true size

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=cv,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
)

start = time.time()
random_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"Combinations tried: {n_iter}  (vs. the full space, which is effectively infinite due to continuous params)")
print(f"Total model fits: {n_iter * cv.get_n_splits()}")
print(f"Time taken: {elapsed:.2f} seconds")
print(f"\nBest hyperparameters: {random_search.best_params_}")
print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"\nFinal test accuracy (evaluated once): {test_score:.4f}")

print("\n--- Compare with 06_Grid_Search/grid_search_example.py ---")
print("Grid Search tried 18 combinations (90 fits) on a small discrete grid.")
print("Random Search tried 20 combinations (100 fits) on a MUCH larger continuous space,")
print("and typically lands very close to Grid Search's best score.")
