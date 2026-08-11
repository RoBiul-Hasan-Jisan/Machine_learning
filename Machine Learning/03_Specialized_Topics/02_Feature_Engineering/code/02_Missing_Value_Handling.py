"""

Strategies for handling missing values, from simplest (drop) to most
information-using (KNN imputation). Every "fit"-style function here is
written to be fit on TRAIN data and applied to TEST data separately,
mirroring how you'd use these in a real pipeline (see the note in each
function) -- fitting on the full dataset before splitting leaks test
information into training.
"""

import numpy as np
import pandas as pd



# Drop-based handling


def drop_missing_rows(df, subset=None, how="any"):
    """Drop rows with missing values. `how='any'` drops if any column in
    `subset` is missing; `how='all'` drops only if every column is missing."""
    return df.dropna(subset=subset, how=how).reset_index(drop=True)


def drop_sparse_columns(df, threshold=0.5):
    """
    Drop columns where more than `threshold` fraction of values are
    missing -- past a certain point, imputation is mostly guessing.
    """
    missing_frac = df.isna().mean()
    keep_cols = missing_frac[missing_frac <= threshold].index
    dropped = missing_frac[missing_frac > threshold].index.tolist()
    return df[keep_cols].copy(), dropped



# Simple statistic imputation (mean / median / mode)


class SimpleImputer:
    """
    Fits a per-column fill value (mean, median, or mode) on training
    data, then applies that same fixed value to any dataset passed to
    `transform` -- including the test set. This is the from-scratch
    equivalent of sklearn's SimpleImputer.
    """

    def __init__(self, strategy="mean"):
        assert strategy in {"mean", "median", "mode"}
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, df, columns):
        for col in columns:
            series = df[col].dropna()
            if self.strategy == "mean":
                self.fill_values_[col] = series.mean()
            elif self.strategy == "median":
                self.fill_values_[col] = series.median()
            else:  # mode
                self.fill_values_[col] = series.mode().iloc[0] if not series.mode().empty else np.nan
        return self

    def transform(self, df):
        df = df.copy()
        for col, value in self.fill_values_.items():
            df[col] = df[col].fillna(value)
        return df

    def fit_transform(self, df, columns):
        return self.fit(df, columns).transform(df)



# KNN imputation


def knn_impute(df, columns, k=5):
    """
    Impute missing numeric values using the mean of the k nearest rows
    (by Euclidean distance over the OTHER numeric columns that are
    present for both rows). This captures information mean/median
    imputation can't: if missingness is related to other observed
    features (MAR), similar rows are a better guess than a global
    statistic.

    Implementation notes:
      - Distance is computed only over columns that are non-missing in
        BOTH rows being compared, so partially-missing rows can still
        contribute to each other's neighbor search.
      - This is O(n^2) in the number of rows -- fine for exploration /
        moderate datasets, not meant for millions of rows without
        indexing (a KD-tree, as sklearn's KNNImputer uses internally).

    Args:
        df: DataFrame with numeric columns (missing values as NaN)
        columns: which columns to impute
        k: number of neighbors to average
    """
    df = df.copy()
    data = df[columns].to_numpy(dtype=float).copy()
    n = len(data)

    for col_idx, col in enumerate(columns):
        missing_rows = np.where(np.isnan(data[:, col_idx]))[0]
        if len(missing_rows) == 0:
            continue

        for row_i in missing_rows:
            dists = np.full(n, np.inf)
            for row_j in range(n):
                if row_j == row_i or np.isnan(data[row_j, col_idx]):
                    continue  # can't borrow from a row that's also missing this column
                # compare only on columns present in BOTH rows
                valid = ~np.isnan(data[row_i]) & ~np.isnan(data[row_j])
                valid[col_idx] = False  # never use the target column itself
                if not valid.any():
                    continue
                diff = data[row_i, valid] - data[row_j, valid]
                dists[row_j] = np.sqrt(np.sum(diff ** 2))

            nearest = np.argsort(dists)[:k]
            nearest = nearest[np.isfinite(dists[nearest])]
            if len(nearest) == 0:
                continue  # no usable neighbors; leave as NaN (fall back to SimpleImputer)
            data[row_i, col_idx] = data[nearest, col_idx].mean()

    result = df.copy()
    result[columns] = data
    return result





def _demo():
   
    print("  MISSING VALUE HANDLING DEMO")
   

    rng = np.random.RandomState(0)
    n = 12
    age = rng.randint(20, 60, n).astype(float)
    income = age * 800 + rng.normal(0, 2000, n)  # income correlated with age

    df = pd.DataFrame({"age": age, "income": income})
    # inject missingness (MAR-ish: income missing more often for younger rows)
    young_idx = np.where(age < 35)[0]
    missing_income_idx = rng.choice(young_idx, size=min(3, len(young_idx)), replace=False)
    df.loc[missing_income_idx, "income"] = np.nan
    df.loc[3, "age"] = np.nan

    print("\nData with missing values:")
    print(df.round(1))

    print(f"\nMissing fraction per column:\n{df.isna().mean().round(2)}")

    mean_imp = SimpleImputer(strategy="mean").fit_transform(df, columns=["age", "income"])
    median_imp = SimpleImputer(strategy="median").fit_transform(df, columns=["age", "income"])
    knn_imp = knn_impute(df, columns=["age", "income"], k=3)

    print("\nMean imputation:")
    print(mean_imp.round(1))
    print("\nMedian imputation:")
    print(median_imp.round(1))
    print("\nKNN imputation (k=3, uses age to inform income fill):")
    print(knn_imp.round(1))

    print("\nNote: KNN fills each missing income using the 3 rows with the")
    print("closest age -- since income correlates with age here, KNN should")
    print("produce more plausible values than the flat global mean/median.")


if __name__ == "__main__":
    _demo()
