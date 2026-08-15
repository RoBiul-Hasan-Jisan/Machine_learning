

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Basic 2-step pipeline ---
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    print("Test accuracy (2-step pipeline):", round(pipe.score(X_test, y_test), 4))

    # Inspect what the model actually sees after preprocessing
    transformed = pipe[:-1].transform(X_test)
    print("First transformed row (should be ~standardized):", transformed[0][:5].round(3))

    # --- Cross-validation on the pipeline (leakage-safe) ---
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print("5-fold CV accuracy: mean =", round(cv_scores.mean(), 4), "std =", round(cv_scores.std(), 4))

    # --- 3-step pipeline with PCA, tuned jointly with the model ---
    pipe3 = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "pca__n_components": [5, 10, 15],
        "model__max_depth": [None, 5, 10],
    }
    search = GridSearchCV(pipe3, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)

    print("\nBest params:", search.best_params_)
    print("Best CV accuracy:", round(search.best_score_, 4))
    print("Test accuracy with best pipeline:", round(search.best_estimator_.score(X_test, y_test), 4))


if __name__ == "__main__":
    main()
