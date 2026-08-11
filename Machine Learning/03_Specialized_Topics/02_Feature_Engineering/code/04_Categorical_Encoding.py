"""

Turn categorical columns into numbers, five different ways. Each encoder
is a small class with fit/transform so it can be fit on TRAIN data and
applied to TEST data without leaking test-set category statistics into
training (critical for target and frequency encoding especially).
"""

import numpy as np
import pandas as pd



# Label Encoding


class LabelEncoder:
    """
    Maps each category to an arbitrary integer (0, 1, 2, ...). Fine for
    tree-based models (a split like "category <= 2" doesn't imply a real
    ordering relationship). Risky for linear/distance-based models,
    which WILL interpret the integers as having magnitude and order.
    """

    def __init__(self):
        self.mapping_ = {}
        self.classes_ = []

    def fit(self, series):
        self.classes_ = sorted(series.dropna().unique().tolist())
        self.mapping_ = {cat: i for i, cat in enumerate(self.classes_)}
        return self

    def transform(self, series):
        # unseen categories (present in test, not train) map to -1
        return series.map(lambda v: self.mapping_.get(v, -1))

    def fit_transform(self, series):
        return self.fit(series).transform(series)



# One-Hot Encoding


class OneHotEncoder:
    """
    Creates one binary column per category. No false ordering implied,
    but column count grows linearly with cardinality -- fine for a
    "color" column with 5 values, problematic for a "user_id"-like
    column with thousands.
    """

    def __init__(self, drop_first=False):
        self.drop_first = drop_first
        self.categories_ = []

    def fit(self, series):
        self.categories_ = sorted(series.dropna().unique().tolist())
        if self.drop_first and self.categories_:
            self.categories_ = self.categories_[1:]  # drop first to avoid multicollinearity
        return self

    def transform(self, series, prefix="cat"):
        cols = {}
        for cat in self.categories_:
            cols[f"{prefix}_{cat}"] = (series == cat).astype(int)
        return pd.DataFrame(cols, index=series.index)

    def fit_transform(self, series, prefix="cat"):
        return self.fit(series).transform(series, prefix)



# Ordinal Encoding


class OrdinalEncoder:
    """
    Like LabelEncoder, but the integer order is EXPLICITLY meaningful
    because you supply it (e.g. low < medium < high), rather than being
    an arbitrary artifact of sort order or first-seen order.
    """

    def __init__(self, order):
        """`order`: list of categories from lowest to highest."""
        self.mapping_ = {cat: i for i, cat in enumerate(order)}

    def transform(self, series):
        return series.map(lambda v: self.mapping_.get(v, np.nan))



# Target Encoding (with out-of-fold fitting to avoid leakage)


class TargetEncoder:
    """
    Replaces each category with the (smoothed) mean target value for
    that category. Powerful for high-cardinality categoricals, but the
    single easiest encoder to leak target information through if you
    fit and transform on the same rows.

    Smoothing blends the category's own mean with the global mean,
    weighted by how many samples that category has -- this prevents a
    rare category (e.g. 1 occurrence) from just memorizing that single
    row's target value.

        smoothed_mean = (count * category_mean + smoothing * global_mean)
                         / (count + smoothing)

    fit_transform() uses K-fold out-of-fold encoding: each row's encoded
    value is computed using the mean from every OTHER fold, never from a
    fold containing that row itself. This is the from-scratch equivalent
    of category_encoders' TargetEncoder with cross-validation.
    """

    def __init__(self, smoothing=10.0, n_folds=5, random_state=42):
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.random_state = random_state
        self.global_mean_ = None
        self.category_means_ = {}

    def fit(self, series, target):
        self.global_mean_ = target.mean()
        stats = target.groupby(series).agg(["mean", "count"])
        self.category_means_ = {
            cat: (row["count"] * row["mean"] + self.smoothing * self.global_mean_)
                 / (row["count"] + self.smoothing)
            for cat, row in stats.iterrows()
        }
        return self

    def transform(self, series):
        return series.map(lambda v: self.category_means_.get(v, self.global_mean_))

    def fit_transform(self, series, target):
        """Out-of-fold fitting: prevents a row's own target from leaking
        into its own encoded value."""
        rng = np.random.RandomState(self.random_state)
        n = len(series)
        fold_ids = rng.randint(0, self.n_folds, size=n)
        encoded = np.zeros(n)

        for fold in range(self.n_folds):
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold

            fold_encoder = TargetEncoder(smoothing=self.smoothing)
            fold_encoder.fit(series[train_mask], target[train_mask])
            encoded[val_mask] = fold_encoder.transform(series[val_mask]).fillna(
                fold_encoder.global_mean_
            )

        # fit the final mapping on ALL data, for use on the real test set later
        self.fit(series, target)
        return pd.Series(encoded, index=series.index)



# Frequency Encoding


class FrequencyEncoder:
    """
    Replaces each category with how often it appears (count or
    proportion) in the training data. No target leakage risk since it
    never looks at the target. Weakness: two different categories with
    the same frequency become indistinguishable to the model.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize
        self.freq_map_ = {}

    def fit(self, series):
        counts = series.value_counts(normalize=self.normalize)
        self.freq_map_ = counts.to_dict()
        return self

    def transform(self, series):
        return series.map(lambda v: self.freq_map_.get(v, 0))

    def fit_transform(self, series):
        return self.fit(series).transform(series)






def _demo():
    
    print("  CATEGORICAL ENCODING DEMO")
   

    df = pd.DataFrame({
        "size": ["small", "medium", "large", "medium", "small", "large", "large"],
        "city": ["NY", "NY", "LA", "SF", "LA", "NY", "SF"],
        "purchased": [0, 1, 1, 0, 0, 1, 1],
    })
    print("\nRaw data:")
    print(df)

    print("\n--- Label Encoding ('city') ---")
    le = LabelEncoder()
    print(le.fit_transform(df["city"]).tolist())

    print("\n--- One-Hot Encoding ('city') ---")
    ohe = OneHotEncoder()
    print(ohe.fit_transform(df["city"], prefix="city"))

    print("\n--- Ordinal Encoding ('size', explicit order) ---")
    oe = OrdinalEncoder(order=["small", "medium", "large"])
    print(oe.transform(df["size"]).tolist())

    print("\n--- Frequency Encoding ('city') ---")
    fe = FrequencyEncoder(normalize=True)
    print(fe.fit_transform(df["city"]).round(3).tolist())

    print("\n--- Target Encoding ('city' -> mean of 'purchased', out-of-fold) ---")
    te = TargetEncoder(smoothing=2.0, n_folds=3, random_state=0)
    encoded = te.fit_transform(df["city"], df["purchased"])
    print(encoded.round(3).tolist())
    print(f"(global mean purchase rate = {df['purchased'].mean():.3f}, rare/small-count")
    print(" categories get pulled toward it by the smoothing term)")


if __name__ == "__main__":
    _demo()
