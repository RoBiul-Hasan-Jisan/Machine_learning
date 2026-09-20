"""
Apriori Algorithm from scratch (pure Python, no libraries needed).

Implements frequent itemset mining (level-wise generate-and-prune using the
Apriori principle) and association rule generation with support/confidence/lift.

Run:
    python apriori_scratch.py
"""
from itertools import combinations


def get_support(itemset, transactions):
    itemset = frozenset(itemset)
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)


def apriori(transactions, min_support=0.3):
    """Returns dict: {frozenset(itemset): support} for all frequent itemsets."""
    transactions = [set(t) for t in transactions]
    n = len(transactions)

    # Pass 1: frequent 1-itemsets
    items = set()
    for t in transactions:
        items.update(t)

    current_level = {}
    for item in items:
        support = get_support({item}, transactions)
        if support >= min_support:
            current_level[frozenset([item])] = support

    all_frequent = dict(current_level)
    k = 2

    while current_level:
        prev_itemsets = list(current_level.keys())
        candidates = set()
        # join step: combine pairs of frequent (k-1)-itemsets whose union has size k
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    candidates.add(union)

        # prune step (Apriori principle): every (k-1)-subset must already be frequent
        pruned_candidates = []
        for cand in candidates:
            subsets = combinations(cand, k - 1)
            if all(frozenset(s) in current_level for s in subsets):
                pruned_candidates.append(cand)

        current_level = {}
        for cand in pruned_candidates:
            support = get_support(cand, transactions)
            if support >= min_support:
                current_level[cand] = support

        all_frequent.update(current_level)
        k += 1

    return all_frequent


def generate_rules(frequent_itemsets, min_confidence=0.5):
    """Given {itemset: support}, generate (antecedent, consequent, support,
    confidence, lift) tuples meeting min_confidence."""
    rules = []
    for itemset, support_xy in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent in combinations(items, r):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                support_x = frequent_itemsets.get(antecedent)
                support_y = frequent_itemsets.get(consequent)
                if support_x is None or support_y is None:
                    continue
                confidence = support_xy / support_x
                if confidence >= min_confidence:
                    lift = confidence / support_y
                    rules.append((antecedent, consequent, support_xy, confidence, lift))
    return sorted(rules, key=lambda r: -r[3])


if __name__ == "__main__":
    transactions = [
        {"bread", "milk"},
        {"bread", "diaper", "beer", "eggs"},
        {"milk", "diaper", "beer", "cola"},
        {"bread", "milk", "diaper", "beer"},
        {"bread", "milk", "diaper", "cola"},
    ]

    frequent = apriori(transactions, min_support=0.4)
    print(f"Found {len(frequent)} frequent itemsets (min_support=0.4):")
    for itemset, support in sorted(frequent.items(), key=lambda kv: -kv[1]):
        print(f"  {set(itemset)}: support={support:.2f}")

    rules = generate_rules(frequent, min_confidence=0.6)
    print(f"\nFound {len(rules)} rules (min_confidence=0.6):")
    for antecedent, consequent, support, confidence, lift in rules:
        print(f"  {set(antecedent)} -> {set(consequent)}  "
              f"(support={support:.2f}, confidence={confidence:.2f}, lift={lift:.2f})")
