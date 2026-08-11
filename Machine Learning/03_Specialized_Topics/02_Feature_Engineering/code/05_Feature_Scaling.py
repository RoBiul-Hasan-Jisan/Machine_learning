"""

Scale numeric features so distance-based and gradient-based models
(KNN, SVM, linear/logistic regression, neural nets, PCA) aren't
dominated by whichever feature happens to have the largest raw units.
Tree-based models are scale-invariant and don't need this.

Every scaler here follows fit/transform: fit on TRAIN data only (learn
the mean/std, min/max, or median/IQR), then apply that SAME fitted
transform to test data. Fitting on the full dataset before splitting
leaks test-set statistics into training.
"""

import numpy as np


class StandardScaler:
    """(x - mean) / std -> mean 0, std 1. The default choice for roughly
    normal-ish features."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid divide-by-zero on constant columns
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """(x - min) / (max - min) -> range [0, 1]. Good when you need a
    bounded range (e.g. neural net inputs) and don't have major outliers
    (a single extreme point compresses everything else toward 0)."""

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        data_range = self.max_ - self.min_
        data_range[data_range == 0] = 1.0
        scaled = (X - self.min_) / data_range
        lo, hi = self.feature_range
        return scaled * (hi - lo) + lo

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    """(x - median) / IQR. Uses statistics that are NOT pulled by extreme
    values, unlike mean/std -- the right choice when your data has
    outliers you want to keep (see 03_Outlier_Detection) but don't want
    dominating the scale."""

    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, X):
        self.median_ = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr_ = q3 - q1
        self.iqr_ = np.where(self.iqr_ == 0, 1.0, self.iqr_)
        return self

    def transform(self, X):
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def l2_normalize_rows(X):
    """
    Scale each ROW (not column) so its vector length (L2 norm) is 1.
    Used when the *direction* of a feature vector matters more than its
    magnitude -- e.g. TF-IDF text vectors, or any cosine-similarity
    based comparison. This is a stateless transform (no fit needed,
    since it operates independently per row).
    """
    norms = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return X / norms



def _demo():
   
    print("  FEATURE SCALING DEMO")
  

    rng = np.random.RandomState(0)
    normal_feature = rng.normal(50, 10, 8)
    with_outlier = np.append(rng.normal(50, 10, 7), 500)  # one big outlier

    X = np.column_stack([normal_feature, with_outlier])
    print("\nRaw data (col 0: clean, col 1: has one big outlier):")
    print(np.round(X, 1))

    std = StandardScaler().fit_transform(X)
    print("\nStandardScaler (mean 0, std 1) -- outlier still dominates col 1's scale:")
    print(np.round(std, 2))

    mm = MinMaxScaler().fit_transform(X)
    print("\nMinMaxScaler ([0,1]) -- outlier compresses every other col-1 value near 0:")
    print(np.round(mm, 2))

    rob = RobustScaler().fit_transform(X)
    print("\nRobustScaler (median/IQR) -- col 1's normal values stay well-spread,")
    print("the outlier doesn't compress everything else:")
    print(np.round(rob, 2))

    rows = np.array([[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]])
    l2 = l2_normalize_rows(rows)
    print("\nL2 row normalization (each row becomes a unit vector):")
    print(np.round(l2, 3))
    print(f"Row norms after normalization: {np.round(np.sqrt((l2**2).sum(axis=1)), 3)}")


if __name__ == "__main__":
    _demo()
