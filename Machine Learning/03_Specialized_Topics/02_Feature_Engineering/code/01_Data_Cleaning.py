"""

Mechanical, pre-statistical cleaning: duplicate removal, inconsistent
value normalization, and dtype conversion. This runs before anything
statistical (imputation, scaling, encoding) touches the data.

"""

import numpy as np
import pandas as pd



# Duplicate removal

def remove_exact_duplicates(df, subset=None, keep="first"):
    """
    Remove exact duplicate rows (or duplicate rows on a subset of columns).

    keep="first" keeps the first occurrence, "last" keeps the last,
    keep=False drops ALL rows that have any duplicate (useful when you
    can't trust which copy is "correct").
    """
    before = len(df)
    cleaned = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def find_near_duplicates(df, column, threshold=0.85):
    """
    Flag pairs of string values in `column` that are likely the same
    entity typed differently (e.g. "New York" vs "new york "), using a
    simple from-scratch normalized Levenshtein-ratio similarity.

    Returns a list of (value_a, value_b, similarity) for pairs above
    `threshold`. This is O(n^2) over *unique* values, so it's meant for
    exploratory cleanup of a categorical column, not large free-text.
    """
    def levenshtein(a, b):
        m, n = len(a), len(b)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        dp[:, 0] = np.arange(m + 1)
        dp[0, :] = np.arange(n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # deletion
                    dp[i, j - 1] + 1,      # insertion
                    dp[i - 1, j - 1] + cost,  # substitution
                )
        return dp[m, n]

    def similarity(a, b):
        a, b = a.strip().lower(), b.strip().lower()
        if not a and not b:
            return 1.0
        dist = levenshtein(a, b)
        return 1 - dist / max(len(a), len(b), 1)

    values = df[column].dropna().astype(str).unique().tolist()
    pairs = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            sim = similarity(values[i], values[j])
            if sim >= threshold and values[i].strip().lower() != values[j].strip().lower():
                pairs.append((values[i], values[j], round(sim, 3)))
    return pairs



# Inconsistent value normalization


def normalize_text_column(series):
    """
    Standardize a categorical/text column: strip whitespace, collapse
    internal whitespace, lowercase. Fixes the most common source of
    accidental category fragmentation ("NY" vs "ny " vs "  NY").
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def apply_value_mapping(series, mapping, default=None):
    """
    Apply an explicit synonym map, e.g. {"nyc": "new york", "ny": "new york"}.
    Values not in the mapping pass through unchanged unless `default` is set.
    """
    if default is None:
        return series.map(lambda v: mapping.get(v, v))
    return series.map(lambda v: mapping.get(v, default))



# Data type conversion


def infer_and_convert_dtypes(df):
    """
    Best-effort dtype inference: try converting each object column to
    numeric, then to datetime, falling back to category if neither works
    and cardinality is low relative to row count.

    Returns (converted_df, report) where report maps column -> chosen dtype.
    """
    df = df.copy()
    report = {}

    for col in df.columns:
        # covers both classic 'object' dtype and pandas' newer 'string'/'str' dtypes
        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            report[col] = str(df[col].dtype)
            continue

        # try numeric
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() > 0.75:  # allow some genuine NaNs / bad rows
            df[col] = numeric
            report[col] = "numeric"
            continue

        # try datetime
        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.notna().mean() > 0.75:
            df[col] = dt
            report[col] = "datetime"
            continue

        # fallback: category if low cardinality
        n_unique = df[col].nunique(dropna=True)
        if n_unique / max(len(df), 1) < 0.5:
            df[col] = df[col].astype("category")
            report[col] = "category"
        else:
            report[col] = "object (left as-is, high cardinality text)"

    return df, report





def _demo():
    
    print("  DATA CLEANING DEMO")
    

    raw = pd.DataFrame({
        "city": ["New York", "new york ", "NYC", "Boston", "boston", "New York"],
        "amount": ["100", "150", "200.5", "not_a_number", "300", "100"],
        "signup_date": ["2024-01-01", "2024-01-02", "2024/01/03", "2024-01-01", "2024-01-05", "2024-01-01"],
    })
    print("\nRaw data:")
    print(raw)

    dedup, removed = remove_exact_duplicates(raw)
    print(f"\nRemoved {removed} exact duplicate row(s).")

    near_dupes = find_near_duplicates(dedup, "city", threshold=0.6)
    print("\nNear-duplicate city values found (similarity >= 0.6):")
    for a, b, sim in near_dupes:
        print(f"  '{a}' ~ '{b}'  (similarity={sim})")

    mapping = {"nyc": "new york", "new york": "new york", "boston": "boston"}
    dedup["city_clean"] = apply_value_mapping(normalize_text_column(dedup["city"]), mapping)
    print("\nAfter normalization + mapping:")
    print(dedup[["city", "city_clean"]])

    converted, report = infer_and_convert_dtypes(dedup[["amount", "signup_date"]])
    print("\nDtype conversion report:")
    for col, dtype in report.items():
        print(f"  {col}: {dtype}")
    print("\nConverted dtypes:")
    print(converted.dtypes)
    print("\n'amount' after conversion (note the unparseable value became NaN):")
    print(converted["amount"])


if __name__ == "__main__":
    _demo()
