

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips each column to [Q1 - k*IQR, Q3 + k*IQR], learned from training data."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


class RatioFeature(BaseEstimator, TransformerMixin):
    """Adds a column: numerator / denominator (stateless, no fitting needed)."""

    def __init__(self, numerator_col=0, denominator_col=1):
        self.numerator_col = numerator_col
        self.denominator_col = denominator_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        denom = X[:, self.denominator_col]
        denom_safe = np.where(denom == 0, np.nan, denom)
        ratio = (X[:, self.numerator_col] / denom_safe).reshape(-1, 1)
        ratio = np.nan_to_num(ratio, nan=0.0)
        return np.hstack([X, ratio])


def demo_outlier_clipper_leakage_safety():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(200, 1))
    X[:20] += 50  # inject outliers concentrated near the start (simulates a skewed fold)

    clipper_full = OutlierClipper(k=1.5).fit(X)
    clipper_fold = OutlierClipper(k=1.5).fit(X[50:])  # a fold that excludes the outliers

    print("Bounds fit on FULL data (with outliers): ", clipper_full.upper_.round(2))
    print("Bounds fit on a FOLD (outliers excluded):", clipper_fold.upper_.round(2))
    print("-> different bounds. This is why clip thresholds must be learned inside")
    print("   each CV fold's training data, not once on the full dataset.\n")


def demo_pipeline_with_custom_transformers():
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(0, 1, size=(n, 3))
    X[:, 0] += rng.choice([0, 40], size=n, p=[0.95, 0.05])  # 5% outliers in col 0
    y = (X[:, 0] * 0.1 + X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 1, n) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("clip", OutlierClipper(k=1.5)),
        ("ratio", RatioFeature(numerator_col=0, denominator_col=1)),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    pipe.fit(X_train, y_train)
    print("Test accuracy:", round(pipe.score(X_test, y_test), 4))

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print("CV accuracy: mean =", round(cv_scores.mean(), 4))

    # Confirm get_params exposes tunable hyperparameters with pipeline-compatible names
    param_grid = {"clip__k": [1.0, 1.5, 3.0]}
    search = GridSearchCV(pipe, param_grid, cv=3)
    search.fit(X_train, y_train)
    print("Best clip__k:", search.best_params_["clip__k"])


def demo_function_transformer_equivalence():
    X = np.array([[1.0], [2.0], [3.0]])

    class LogTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.log1p(X)

    class_based = LogTransformer().fit_transform(X)
    function_based = FunctionTransformer(np.log1p, validate=True).fit_transform(X)

    assert np.allclose(class_based, function_based)
    print("\nClass-based and FunctionTransformer outputs match:", class_based.ravel())


if __name__ == "__main__":
    print("=== OutlierClipper leakage-safety demo ===")
    demo_outlier_clipper_leakage_safety()

    print("=== Pipeline with custom transformers ===")
    demo_pipeline_with_custom_transformers()

    demo_function_transformer_equivalence()
