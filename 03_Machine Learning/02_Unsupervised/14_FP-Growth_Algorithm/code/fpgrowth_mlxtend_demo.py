"""
FP-Growth with the `mlxtend` library, plus a timing comparison against
Apriori on a larger synthetic transaction dataset (illustrating FP-Growth's
"fewer database scans" advantage).

Run:
    pip install mlxtend pandas
    python fpgrowth_mlxtend_demo.py

Falls back to the from-scratch implementations if mlxtend isn't installed.
"""
import random
import time
import pandas as pd


def make_synthetic_transactions(n_transactions=500, n_items=20, random_state=0):
    rng = random.Random(random_state)
    items = [f"item_{i}" for i in range(n_items)]
    # skew item frequencies (like real retail data: a few popular items,
    # a long tail of rare ones) so FP-Growth's compression advantage shows
    weights = [1.0 / (rank + 1) for rank in range(n_items)]
    transactions = []
    for _ in range(n_transactions):
        basket_size = rng.randint(2, 6)
        basket = set(rng.choices(items, weights=weights, k=basket_size))
        transactions.append(basket)
    return transactions


def one_hot_encode(transactions):
    all_items = sorted({item for t in transactions for item in t})
    rows = [{item: (item in t) for item in all_items} for t in transactions]
    return pd.DataFrame(rows)


def main():
    transactions = make_synthetic_transactions()
    df = one_hot_encode(transactions)

    try:
        from mlxtend.frequent_patterns import fpgrowth, apriori

        t0 = time.perf_counter()
        fp_result = fpgrowth(df, min_support=0.05, use_colnames=True)
        t_fp = time.perf_counter() - t0

        t0 = time.perf_counter()
        ap_result = apriori(df, min_support=0.05, use_colnames=True)
        t_ap = time.perf_counter() - t0

        print(f"mlxtend fpgrowth: {len(fp_result)} itemsets in {t_fp*1000:.2f} ms")
        print(f"mlxtend apriori:  {len(ap_result)} itemsets in {t_ap*1000:.2f} ms")

    except ImportError:
        print("mlxtend not installed -- falling back to from-scratch implementations.\n")
        import sys, os
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        sys.path.insert(0, os.path.join(here, "..", "..", "Apriori Algorithm", "code"))
        from fpgrowth_scratch import fpgrowth
        from apriori_scratch import apriori

        t0 = time.perf_counter()
        fp_result = fpgrowth(transactions, min_support=0.05)
        t_fp = time.perf_counter() - t0

        t0 = time.perf_counter()
        ap_result = apriori(transactions, min_support=0.05)
        t_ap = time.perf_counter() - t0

        print(f"from-scratch fpgrowth: {len(fp_result)} itemsets in {t_fp*1000:.2f} ms")
        print(f"from-scratch apriori:  {len(ap_result)} itemsets in {t_ap*1000:.2f} ms")
        print("Same itemsets found:", set(fp_result.keys()) == set(ap_result.keys()))


if __name__ == "__main__":
    main()
