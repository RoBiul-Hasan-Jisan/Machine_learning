"""
FP-Growth Algorithm from scratch (pure Python).

Builds a compressed FP-tree from the transaction database (after two scans:
one to count item frequencies, one to insert reordered transactions), then
recursively mines it via conditional pattern bases / conditional FP-trees --
no candidate generation, unlike Apriori.

Run:
    python fpgrowth_scratch.py
"""
from collections import defaultdict


class FPNode:
    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None  # points to the next node in the tree with the same item


def build_tree(transactions_with_counts, min_support_count):
    """transactions_with_counts: list of (transaction_items_iterable, count).
    Returns (root, header_table) or (None, None) if nothing is frequent.
    header_table: {item: [support_count, first_node_or_None]}
    """
    item_counts = defaultdict(int)
    for items, count in transactions_with_counts:
        for item in items:
            item_counts[item] += count

    frequent_items = {item: c for item, c in item_counts.items() if c >= min_support_count}
    if not frequent_items:
        return None, None

    header_table = {item: [count, None] for item, count in frequent_items.items()}

    root = FPNode(None, 0, None)

    def order_key(item):
        return (-frequent_items[item], item)  # descending frequency, then item name for stability

    for items, count in transactions_with_counts:
        ordered_items = sorted((i for i in items if i in frequent_items), key=order_key)
        if not ordered_items:
            continue
        current = root
        for item in ordered_items:
            if item in current.children:
                current.children[item].count += count
            else:
                new_node = FPNode(item, count, current)
                current.children[item] = new_node
                # link into header table's node-link chain
                if header_table[item][1] is None:
                    header_table[item][1] = new_node
                else:
                    node = header_table[item][1]
                    while node.node_link is not None:
                        node = node.node_link
                    node.node_link = new_node
            current = current.children[item]

    return root, header_table


def ascend_path(node):
    """Return the list of items from node's parent up to (excluding) the root."""
    path = []
    node = node.parent
    while node is not None and node.item is not None:
        path.append(node.item)
        node = node.parent
    return path


def find_conditional_pattern_base(header_table, item):
    """All prefix paths (with counts) leading to occurrences of `item`."""
    node = header_table[item][1]
    pattern_base = []
    while node is not None:
        path = ascend_path(node)
        if path:
            pattern_base.append((path, node.count))
        node = node.node_link
    return pattern_base


def fp_growth_mine(header_table, min_support_count, prefix, frequent_itemsets):
    # mine items from least frequent to most frequent (standard convention;
    # doesn't affect correctness, just recursion structure)
    items_by_ascending_support = sorted(header_table.keys(), key=lambda i: header_table[i][0])

    for item in items_by_ascending_support:
        support_count = header_table[item][0]
        new_pattern = prefix + [item]
        frequent_itemsets[frozenset(new_pattern)] = support_count

        pattern_base = find_conditional_pattern_base(header_table, item)
        cond_tree, cond_header = build_tree(pattern_base, min_support_count)
        if cond_header:
            fp_growth_mine(cond_header, min_support_count, new_pattern, frequent_itemsets)


def fpgrowth(transactions, min_support=0.3):
    """transactions: list of iterables of items. Returns {frozenset: support (fraction)}."""
    n = len(transactions)
    min_support_count = min_support * n
    transactions_with_counts = [(t, 1) for t in transactions]

    root, header_table = build_tree(transactions_with_counts, min_support_count)
    if header_table is None:
        return {}

    frequent_counts = {}
    fp_growth_mine(header_table, min_support_count, [], frequent_counts)

    return {itemset: count / n for itemset, count in frequent_counts.items()}


if __name__ == "__main__":
    transactions = [
        {"bread", "milk"},
        {"bread", "diaper", "beer", "eggs"},
        {"milk", "diaper", "beer", "cola"},
        {"bread", "milk", "diaper", "beer"},
        {"bread", "milk", "diaper", "cola"},
    ]

    frequent = fpgrowth(transactions, min_support=0.4)
    print(f"FP-Growth found {len(frequent)} frequent itemsets (min_support=0.4):")
    for itemset, support in sorted(frequent.items(), key=lambda kv: -kv[1]):
        print(f"  {set(itemset)}: support={support:.2f}")

    # cross-check against Apriori and Eclat
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "Apriori Algorithm", "code"))
    sys.path.insert(0, os.path.join(here, "..", "..", "Eclat Algorithm", "code"))
    from apriori_scratch import apriori
    from eclat_scratch import eclat

    apriori_result = apriori(transactions, min_support=0.4)
    eclat_result = eclat(transactions, min_support=0.4)

    print(f"\nApriori found {len(apriori_result)} itemsets")
    print(f"Eclat found {len(eclat_result)} itemsets")
    print(f"FP-Growth found {len(frequent)} itemsets")
    print("All three match:", frequent == apriori_result == eclat_result)
