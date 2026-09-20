"""
Eclat Algorithm from scratch (pure Python).

Vertical (TID-set) data representation + depth-first frequent itemset
mining via set intersection.

Run:
    python eclat_scratch.py
"""
import time


def build_tid_sets(transactions):
    """transactions: list of sets of items. Returns {item: set(transaction_indices)}."""
    tid_sets = {}
    for tid, t in enumerate(transactions):
        for item in t:
            tid_sets.setdefault(item, set()).add(tid)
    return tid_sets


def eclat(transactions, min_support=0.3):
    """Returns dict: {frozenset(itemset): support}."""
    n = len(transactions)
    tid_sets = build_tid_sets(transactions)

    # keep only frequent single items, sorted for a consistent extension order
    frequent_items = {
        item: tids for item, tids in tid_sets.items()
        if len(tids) / n >= min_support
    }
    sorted_items = sorted(frequent_items.keys())

    all_frequent = {}
    for item in sorted_items:
        all_frequent[frozenset([item])] = len(frequent_items[item]) / n

    def extend(prefix_itemset, prefix_tids, candidate_items):
        """candidate_items: sorted list of items that can extend prefix_itemset
        (all items lexicographically after the last item added to the prefix)."""
        for idx, item in enumerate(candidate_items):
            new_tids = prefix_tids & frequent_items[item]
            support = len(new_tids) / n
            if support >= min_support:
                new_itemset = prefix_itemset | {item}
                all_frequent[frozenset(new_itemset)] = support
                # recurse only into items that come after `item` in the ordering
                extend(new_itemset, new_tids, candidate_items[idx + 1:])

    for idx, item in enumerate(sorted_items):
        extend({item}, frequent_items[item], sorted_items[idx + 1:])

    return all_frequent


if __name__ == "__main__":
    transactions = [
        {"bread", "milk"},
        {"bread", "diaper", "beer", "eggs"},
        {"milk", "diaper", "beer", "cola"},
        {"bread", "milk", "diaper", "beer"},
        {"bread", "milk", "diaper", "cola"},
    ]

    frequent = eclat(transactions, min_support=0.4)
    print(f"Eclat found {len(frequent)} frequent itemsets (min_support=0.4):")
    for itemset, support in sorted(frequent.items(), key=lambda kv: -kv[1]):
        print(f"  {set(itemset)}: support={support:.2f}")

    # --- cross-check against Apriori, and a rough timing comparison ---
    import sys
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    apriori_path = os.path.join(here, "..", "..", "Apriori Algorithm", "code")
    sys.path.insert(0, apriori_path)
    from apriori_scratch import apriori

    t0 = time.perf_counter()
    eclat_result = eclat(transactions, min_support=0.4)
    t_eclat = time.perf_counter() - t0

    t0 = time.perf_counter()
    apriori_result = apriori(transactions, min_support=0.4)
    t_apriori = time.perf_counter() - t0

    print(f"\nEclat found {len(eclat_result)} itemsets in {t_eclat*1000:.3f} ms")
    print(f"Apriori found {len(apriori_result)} itemsets in {t_apriori*1000:.3f} ms")
    print("Results match:", eclat_result == apriori_result)
