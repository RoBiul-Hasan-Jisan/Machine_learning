"""

Detect (and optionally cap) outliers. Detection != automatic deletion --
these functions return a mask/flag so you can decide whether an outlier
is a data error or a genuine rare event worth keeping (or modeling
separately).
"""

import numpy as np



# IQR method


def iqr_outliers(values, k=1.5):
    """
    Flag values outside [Q1 - k*IQR, Q3 + k*IQR]. The classic boxplot
    rule. Robust to non-normal distributions since it's based on
    quantiles, not mean/std.

    Returns: (mask, lower_bound, upper_bound) where mask[i]=True means
    values[i] is an outlier.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (values < lower) | (values > upper)
    return mask, lower, upper



# Z-score method


def zscore_outliers(values, threshold=3.0):
    """
    Flag values whose |z-score| exceeds `threshold`. Assumes roughly
    normal data. Note: mean and std are themselves pulled by outliers
    (not robust), so extreme outliers can mask moderate ones -- IQR or
    Winsorization degrade more gracefully when outliers are severe.
    """
    mean, std = values.mean(), values.std()
    if std == 0:
        return np.zeros(len(values), dtype=bool), np.zeros(len(values))
    z = (values - mean) / std
    mask = np.abs(z) > threshold
    return mask, z



# Winsorization


def winsorize(values, lower_pct=1, upper_pct=99):
    """
    Cap (not remove) values below the `lower_pct` percentile or above
    the `upper_pct` percentile. Keeps every row (sample size preserved)
    while reducing the leverage of extreme values -- useful when you
    want to keep a feature but don't want a handful of extreme points
    dominating a mean, a regression coefficient, or a distance metric.
    """
    lower = np.percentile(values, lower_pct)
    upper = np.percentile(values, upper_pct)
    return np.clip(values, lower, upper)



# Isolation Forest (simplified, from scratch)


class _IsolationTree:
    """A single random partitioning tree used by IsolationForest."""

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.is_leaf = True

    def fit(self, X, depth, rng):
        self.size = len(X)
        if depth >= self.max_depth or len(X) <= 1:
            self.is_leaf = True
            return self

        n_features = X.shape[1]
        self.split_feature = rng.randint(0, n_features)
        col = X[:, self.split_feature]
        col_min, col_max = col.min(), col.max()
        if col_min == col_max:
            self.is_leaf = True
            return self

        self.split_value = rng.uniform(col_min, col_max)
        self.is_leaf = False

        left_mask = col < self.split_value
        self.left = _IsolationTree(self.max_depth).fit(X[left_mask], depth + 1, rng)
        self.right = _IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1, rng)
        return self

    def path_length(self, x, depth=0):
        if self.is_leaf:
            # add an adjustment for the average path length of unbuilt
            # subtree size (standard isolation forest correction term)
            return depth + _c(self.size)
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


def _c(n):
    """Average path length of an unsuccessful search in a Binary Search
    Tree of n nodes -- the standard normalization constant used by
    Isolation Forest so path lengths are comparable across tree/sample
    sizes."""
    if n <= 1:
        return 0.0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n  # 0.577... = Euler-Mascheroni constant


class IsolationForest:
    """
    From-scratch Isolation Forest for outlier detection.

    Core idea: outliers are "few and different," so random partitioning
    isolates them in FEWER splits than normal points, which sit in dense
    regions and take many splits to separate from their neighbors.
    Average path length across many random trees, converted to an
    anomaly score, gives a distribution-free outlier measure that works
    in high dimensions -- unlike IQR/Z-score, which are naturally 1D.
    """

    def __init__(self, n_trees=100, sample_size=256, random_state=42):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        sample_size = min(self.sample_size, n)
        max_depth = int(np.ceil(np.log2(max(sample_size, 2))))

        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=sample_size, replace=False)
            tree = _IsolationTree(max_depth).fit(X[idx], depth=0, rng=rng)
            self.trees_.append(tree)
        self._c_n = _c(sample_size)
        return self

    def score_samples(self, X):
        """
        Returns anomaly scores in (0, 1]. Score close to 1 = likely
        outlier (short average path length). Score well below 0.5 =
        likely normal (long average path length, deeply embedded in a
        dense region).
        """
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            avg_path = np.mean([tree.path_length(x) for tree in self.trees_])
            scores[i] = 2 ** (-avg_path / self._c_n) if self._c_n > 0 else 0.5
        return scores

    def predict(self, X, threshold=0.6):
        """Return a boolean mask: True where the anomaly score exceeds `threshold`."""
        return self.score_samples(X) > threshold






def _demo():
    
    print("  OUTLIER DETECTION DEMO")
  

    rng = np.random.RandomState(1)
    normal = rng.normal(50, 5, 95)
    outliers = np.array([5, 8, 95, 98, 110])
    values = np.concatenate([normal, outliers])

    mask_iqr, lower, upper = iqr_outliers(values)
    print(f"\nIQR method: bounds = [{lower:.2f}, {upper:.2f}]")
    print(f"  Flagged {mask_iqr.sum()} outliers, values: {np.round(values[mask_iqr], 1)}")

    mask_z, z = zscore_outliers(values, threshold=3.0)
    print(f"\nZ-score method (threshold=3): flagged {mask_z.sum()} outliers")
    print(f"  values: {np.round(values[mask_z], 1)}")

    winsorized = winsorize(values, lower_pct=2, upper_pct=98)
    print(f"\nWinsorization (2nd/98th percentile): min/max before = "
          f"{values.min():.1f}/{values.max():.1f}, after = "
          f"{winsorized.min():.1f}/{winsorized.max():.1f}")

    # Isolation Forest on a 2D dataset (multivariate outliers IQR/Z-score can't see)
    normal_2d = rng.normal(0, 1, (100, 2))
    outliers_2d = rng.uniform(-8, 8, (8, 2))
    X = np.vstack([normal_2d, outliers_2d])

    iso = IsolationForest(n_trees=100, sample_size=64, random_state=42).fit(X)
    scores = iso.score_samples(X)
    mask_iso = iso.predict(X, threshold=0.6)
    print(f"\nIsolation Forest on 2D data: flagged {mask_iso.sum()} / {len(X)} points as outliers")
    print(f"  (planted {len(outliers_2d)} synthetic outliers; top 5 anomaly scores below)")
    top5 = np.argsort(-scores)[:5]
    for i in top5:
        print(f"  point {i}: score={scores[i]:.3f}, coords={np.round(X[i], 2)}")


if __name__ == "__main__":
    _demo()
