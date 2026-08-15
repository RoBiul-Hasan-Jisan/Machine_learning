
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)


def leaky_vs_correct_split():
    X, y = make_classification(n_samples=500, n_features=10, weights=[0.9, 0.1], random_state=42)

    # --- WRONG: fit scaler on all data before splitting ---
    scaler_leaky = StandardScaler().fit(X)
    X_scaled_leaky = scaler_leaky.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_leaky, y, test_size=0.25, random_state=42, stratify=y
    )
    model_leaky = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    leaky_acc = model_leaky.score(X_test, y_test)

    # --- RIGHT: split first, fit scaler on train only ---
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler_correct = StandardScaler().fit(X_train_r)
    X_train_scaled = scaler_correct.transform(X_train_r)
    X_test_scaled = scaler_correct.transform(X_test_r)
    model_correct = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train_r)
    correct_acc = model_correct.score(X_test_scaled, y_test_r)

    print(f"Leaky-split test accuracy:   {leaky_acc:.4f}")
    print(f"Correct-split test accuracy: {correct_acc:.4f}")
    print("(On simple datasets the gap may be small, but the leaky number")
    print(" is still an invalid estimate of real generalization.)\n")


def stratified_demo():
    X, y = make_classification(n_samples=500, n_features=5, weights=[0.95, 0.05], random_state=1)

    _, _, _, y_test_strat = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    _, _, _, y_test_random = train_test_split(X, y, test_size=0.2, random_state=1, stratify=None)

    print("Full dataset positive rate: ", round(y.mean(), 4))
    print("Stratified test positive rate:", round(y_test_strat.mean(), 4))
    print("Random test positive rate:    ", round(y_test_random.mean(), 4), "\n")


def group_split_demo():
    n_customers = 50
    rows_per_customer = 5
    customer_id = np.repeat(np.arange(n_customers), rows_per_customer)
    n_rows = len(customer_id)
    X = rng.normal(size=(n_rows, 3))
    y = rng.integers(0, 2, size=n_rows)

    # naive random split: same customer can land in both sets
    train_idx, test_idx = train_test_split(np.arange(n_rows), test_size=0.2, random_state=0)
    overlap_random = set(customer_id[train_idx]) & set(customer_id[test_idx])

    # group-aware split: no customer appears in both sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx_g, test_idx_g = next(gss.split(X, y, groups=customer_id))
    overlap_group = set(customer_id[train_idx_g]) & set(customer_id[test_idx_g])

    print(f"Random split customer overlap: {len(overlap_random)} customers")
    print(f"GroupShuffleSplit customer overlap: {len(overlap_group)} customers")


if __name__ == "__main__":
    print("=== Leaky vs correct split ===")
    leaky_vs_correct_split()

    print("=== Stratified split ===")
    stratified_demo()

    print("=== Group split ===")
    group_split_demo()
