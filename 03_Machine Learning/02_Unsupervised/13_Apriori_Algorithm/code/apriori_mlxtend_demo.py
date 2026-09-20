"""
Apriori with the `mlxtend` library (the standard, well-tested library
implementation) — one-hot encode transactions, mine frequent itemsets,
generate rules with support/confidence/lift.

Run:
    pip install mlxtend pandas
    python apriori_mlxtend_demo.py

If mlxtend isn't installed, this script falls back to a small pandas-only
re-implementation of the same idea so it still runs end-to-end.
"""
import pandas as pd


TRANSACTIONS = [
    ["bread", "milk"],
    ["bread", "diaper", "beer", "eggs"],
    ["milk", "diaper", "beer", "cola"],
    ["bread", "milk", "diaper", "beer"],
    ["bread", "milk", "diaper", "cola"],
]


def one_hot_encode(transactions):
    all_items = sorted({item for t in transactions for item in t})
    rows = [{item: (item in t) for item in all_items} for t in transactions]
    return pd.DataFrame(rows)


def main():
    df = one_hot_encode(TRANSACTIONS)

    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

        print("=== mlxtend apriori ===")
        print(frequent_itemsets.sort_values("support", ascending=False).to_string(index=False))
        print("\n=== mlxtend association_rules ===")
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(rules[cols].sort_values("confidence", ascending=False).to_string(index=False))

    except ImportError:
        print("mlxtend not installed -- falling back to a minimal pandas-only apriori.\n")
        from apriori_scratch import apriori as apriori_scratch, generate_rules

        transactions_as_sets = [set(t) for t in TRANSACTIONS]
        frequent = apriori_scratch(transactions_as_sets, min_support=0.4)
        rules = generate_rules(frequent, min_confidence=0.6)

        print("Frequent itemsets:")
        for itemset, support in sorted(frequent.items(), key=lambda kv: -kv[1]):
            print(f"  {set(itemset)}: support={support:.2f}")

        print("\nRules:")
        for antecedent, consequent, support, confidence, lift in rules:
            print(f"  {set(antecedent)} -> {set(consequent)}  "
                  f"(support={support:.2f}, confidence={confidence:.2f}, lift={lift:.2f})")


if __name__ == "__main__":
    main()
