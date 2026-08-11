

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def single_split_variance(X, y, n_trials=10):
    scores = []
    for seed in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        pipe = build_pipeline().fit(X_train, y_train)
        scores.append(pipe.score(X_test, y_test))
    return np.array(scores)


def kfold_scores(X, y, n_splits=5):
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")


def grid_search_demo(X_train, y_train):
    pipe = build_pipeline()
    param_grid = {"model__C": [0.01, 0.1, 1, 10, 100]}
    search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def nested_cv_demo(X_train, y_train):
    pipe = build_pipeline()
    param_grid = {"model__C": [0.01, 0.1, 1, 10, 100]}
    inner = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
    nested_scores = cross_val_score(inner, X_train, y_train, cv=5, scoring="accuracy")
    return nested_scores


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    print("=== Single-split variance (10 different random seeds) ===")
    single_scores = single_split_variance(X_train, y_train)
    print(f"scores: {single_scores.round(4)}")
    print(f"mean={single_scores.mean():.4f}  std={single_scores.std():.4f}\n")

    print("=== 5-fold CV (one call, more stable estimate) ===")
    cv_scores = kfold_scores(X_train, y_train)
    print(f"scores: {cv_scores.round(4)}")
    print(f"mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}\n")

    print("=== GridSearchCV over model__C ===")
    search = grid_search_demo(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV score:", round(search.best_score_, 4))
    print("Test score of refit best pipeline:", round(search.best_estimator_.score(X_test, y_test), 4), "\n")

    print("=== Nested CV (honest estimate of the tuning process) ===")
    nested_scores = nested_cv_demo(X_train, y_train)
    print(f"nested scores: {nested_scores.round(4)}")
    print(f"nested mean={nested_scores.mean():.4f}  (compare to GridSearchCV.best_score_ above)")


if __name__ == "__main__":
    main()
