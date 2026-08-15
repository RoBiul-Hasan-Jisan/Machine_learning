"""

Engineer new features rather than just reshaping existing ones. This is
usually the highest-leverage step in the whole pipeline, and the one
generic libraries can't automate for you (domain knowledge lives here).
"""

import numpy as np
import pandas as pd


# Date & time features

def date_time_features(dates):
    """
    Extract calendar features from a datetime series, including cyclical
    (sin/cos) encodings for hour and day-of-week so that, e.g., 11 PM and
    midnight are close together numerically -- a plain integer encoding
    (0..23) would treat them as maximally far apart, which is wrong for
    any cyclical quantity.
    """
    dates = pd.to_datetime(dates)
    out = pd.DataFrame(index=dates.index if hasattr(dates, "index") else None)

    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["day"] = dates.dt.day
    out["day_of_week"] = dates.dt.dayofweek  # 0=Monday
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour"] = dates.dt.hour

    # cyclical encodings: preserve "closeness" across the wrap-around point
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    return out


def days_since(dates, reference_date):
    """Days elapsed between each date and a fixed reference (e.g. a
    known event, a campaign launch, or 'today') -- turns a date into a
    single ordered numeric feature a model can use directly."""
    dates = pd.to_datetime(dates)
    reference_date = pd.to_datetime(reference_date)
    return (dates - reference_date).dt.days


# Interaction features

def interaction_features(df, pairs, ops=("multiply", "ratio", "difference")):
    """
    Create interaction features from explicit column pairs -- unlike
    Section 06's polynomial_features (which tries all pairs blindly),
    this lets you target interactions you have a domain reason to
    believe matter (e.g. price / square_feet, not every possible pair).

    Args:
        df: source DataFrame
        pairs: list of (col_a, col_b) tuples
        ops: which interaction types to compute for each pair
    """
    out = pd.DataFrame(index=df.index)
    for a, b in pairs:
        if "multiply" in ops:
            out[f"{a}_x_{b}"] = df[a] * df[b]
        if "ratio" in ops:
            out[f"{a}_per_{b}"] = df[a] / df[b].replace(0, np.nan)
        if "difference" in ops:
            out[f"{a}_minus_{b}"] = df[a] - df[b]
    return out


# Aggregation features

def aggregation_features(df, group_col, value_col, aggs=("mean", "std", "count", "max")):
    """
    Compute group-level statistics and join them back onto every row in
    that group -- e.g. each transaction row gets the customer's average
    purchase amount, turning relational/grouped structure into row-level
    signal a standard tabular model can use.

    IMPORTANT LEAKAGE NOTE: if `value_col` is itself derived from the
    target (or the target itself), computing this aggregation using ALL
    rows (including the current row) leaks that row's own outcome into
    its own feature. For target-derived aggregations, compute the group
    statistic using only OTHER rows (leave-one-out) or only past rows
    (if there's a time ordering) -- the same principle as target
    encoding's out-of-fold fitting in Section 04.
    """
    agg_df = df.groupby(group_col)[value_col].agg(list(aggs))
    agg_df.columns = [f"{value_col}_{group_col}_{a}" for a in aggs]
    return df.merge(agg_df, left_on=group_col, right_index=True, how="left")


def leave_one_out_aggregation(df, group_col, value_col, agg="mean"):
    """
    Leak-safe version of aggregation_features for target-derived columns:
    each row's aggregated value is computed using every OTHER row in its
    group, never itself.
    """
    group_sum = df.groupby(group_col)[value_col].transform("sum")
    group_count = df.groupby(group_col)[value_col].transform("count")

    if agg == "mean":
        loo = (group_sum - df[value_col]) / (group_count - 1).replace(0, np.nan)
    elif agg == "sum":
        loo = group_sum - df[value_col]
    else:
        raise ValueError("agg must be 'mean' or 'sum' for this leave-one-out helper")

    return loo


# Binning

def equal_width_bins(values, n_bins=4, labels=None):
    """Divide the value range into n_bins equal-WIDTH intervals."""
    return pd.cut(values, bins=n_bins, labels=labels)


def equal_freq_bins(values, n_bins=4, labels=None):
    """Divide into n_bins intervals with equal COUNTS per bin (quantile
    binning) -- useful when the raw distribution is skewed and
    equal-width bins would leave most points in one or two bins."""
    return pd.qcut(values, q=n_bins, labels=labels, duplicates="drop")


def custom_bins(values, edges, labels=None):
    """Divide using explicit domain-meaningful edges, e.g. age groups
    [0, 18, 35, 60, 120] -> child / young_adult / adult / senior."""
    return pd.cut(values, bins=edges, labels=labels)


# Domain-specific features (worked examples)

def bmi(weight_kg, height_m):
    """Body Mass Index: weight / height^2. A textbook example of a
    domain-specific ratio feature that's far more predictive than
    weight and height as separate raw columns."""
    return weight_kg / (height_m ** 2)


def debt_to_income_ratio(total_debt, annual_income):
    """Common financial-domain feature: how leveraged is this entity
    relative to its income. Directly interpretable, often more
    predictive than either raw column alone in credit-risk models."""
    return total_debt / annual_income.replace(0, np.nan)


def rfm_features(df, customer_col, date_col, amount_col, reference_date=None):
    """
    RFM (Recency, Frequency, Monetary) features -- a classic
    domain-specific feature set from marketing/customer analytics:
      - Recency: days since the customer's most recent transaction
      - Frequency: how many transactions the customer has made
      - Monetary: total (or average) amount spent
    """
    if reference_date is None:
        reference_date = pd.to_datetime(df[date_col]).max()
    else:
        reference_date = pd.to_datetime(reference_date)

    grouped = df.groupby(customer_col).agg(
        recency=(date_col, lambda d: (reference_date - pd.to_datetime(d).max()).days),
        frequency=(date_col, "count"),
        monetary=(amount_col, "sum"),
    )
    return grouped




def _demo():
   
    print("  FEATURE CREATION DEMO")


    df = pd.DataFrame({
        "customer": ["A", "A", "B", "B", "B", "C"],
        "order_date": pd.to_datetime([
            "2024-01-05", "2024-02-10", "2024-01-15", "2024-01-20", "2024-03-01", "2024-02-28"
        ]),
        "amount": [50.0, 75.0, 20.0, 30.0, 40.0, 100.0],
        "weight_kg": [70, 65, 90, 55, 80, 60],
        "height_m": [1.75, 1.60, 1.85, 1.65, 1.78, 1.70],
    })
    print("\nRaw data:")
    print(df)

    dt_feats = date_time_features(df["order_date"])
    print("\nDate/time features (first few columns):")
    print(dt_feats[["year", "month", "day_of_week", "is_weekend", "day_of_week_sin"]].round(2))

    inter = interaction_features(df, pairs=[("amount", "weight_kg")], ops=("ratio",))
    print("\nInteraction feature (amount per kg, arbitrary example pairing):")
    print(inter.round(3))

    agg = aggregation_features(df, group_col="customer", value_col="amount", aggs=("mean", "count"))
    print("\nAggregation features (per-customer amount stats joined back):")
    print(agg[["customer", "amount", "amount_customer_mean", "amount_customer_count"]])

    loo = leave_one_out_aggregation(df, group_col="customer", value_col="amount")
    print("\nLeave-one-out mean amount (leak-safe version -- excludes each row's own amount):")
    print(loo.round(2))

    df["age"] = [25, 40, 65, 30, 50, 22]
    bins = custom_bins(df["age"], edges=[0, 30, 50, 120], labels=["young", "middle", "senior"])
    print("\nCustom age bins:")
    print(pd.DataFrame({"age": df["age"], "bin": bins}))

    df["bmi"] = bmi(df["weight_kg"], df["height_m"])
    print("\nDomain-specific feature (BMI):")
    print(df[["weight_kg", "height_m", "bmi"]].round(2))

    rfm = rfm_features(df, customer_col="customer", date_col="order_date", amount_col="amount")
    print("\nRFM features (recency in days, frequency, monetary total):")
    print(rfm)


if __name__ == "__main__":
    _demo()
