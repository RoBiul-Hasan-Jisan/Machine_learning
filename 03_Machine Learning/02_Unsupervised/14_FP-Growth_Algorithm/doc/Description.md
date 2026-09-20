# FP-Growth Algorithm



##  Motivation

Apriori's dominant cost is repeated full-database scanning: the number of
scans equals the length of the longest frequent itemset, since each level
of the level-wise search requires a separate pass to evaluate that level's
candidates (see `Apriori Algorithm/doc/Description.md`, Section 6).
FP-Growth addresses this cost directly by compressing the transaction
database into a single in-memory tree structure — the **FP-tree** — after
exactly two scans, and mining that structure recursively rather than
generating and testing candidates against the raw data.

##  The FP-Tree: Construction

**Input:** transaction database $D$, threshold `min_support`.

1. **First scan.** Compute the support of every individual item.
2. **Filtering and ordering.** Discard items not meeting `min_support`.
   Within each transaction, sort the remaining items by descending global
   frequency. This ordering is the structural basis for the tree's
   compression: transactions that share their most frequent items will
   share a path prefix in the tree, and sorting by descending frequency
   maximizes the length of prefix shared across transactions, since the
   items most likely to recur across many transactions are placed first.
3. **Second scan.** Insert each filtered, reordered transaction into a
   shared prefix tree, rooted at a null node:
   - Traverse the tree from the root, following the transaction's items in
     order.
   - If a child node for the current item already exists at the current
     position, increment its count.
   - Otherwise, create a new child node with count 1 and continue from
     there.
4. **Header table.** Maintain, for each frequent item, a pointer to its
   first occurrence in the tree. Each node additionally holds a node-link
   to the next node in the tree representing the same item. Together, the
   header table and node-links allow every occurrence of a given item to
   be enumerated directly, without re-scanning the transaction data.

**Compression property.** Because transactions sharing a frequent prefix
reuse the same tree path — incrementing existing node counts rather than
creating new nodes — the FP-tree's size is bounded by the degree of shared
structure across transactions rather than by the raw transaction count.
For correlated, skewed-frequency data typical of retail transactions, this
produces a tree substantially smaller than the original transaction list.

##  Mining the FP-Tree

Mining proceeds by a divide-and-conquer recursion over items, processed
from least frequent to most frequent — an order chosen specifically to
keep each recursive sub-problem as small as possible.

**Conditional pattern base.** For an item $i$, the conditional pattern
base is the set of all prefix paths in the tree that terminate at a node
for $i$, obtained by following $i$'s node-links from the header table.
Each such path represents a transaction (or set of transactions) that
co-occurs with $i$, restricted to the portion of the transaction preceding
$i$ in frequency order.

**Conditional FP-tree.** From the conditional pattern base, a smaller
conditional FP-tree is constructed using the identical procedure described
in Section 2. This tree represents the sub-problem "what is frequent
among items that co-occur with $i$."

**Recursion.** The conditional FP-tree for $i$ is mined by the same
procedure, recursively, until the recursion terminates at single-item
conditional trees. Every frequent itemset containing $i$ is obtained by
combining $i$ with the frequent itemsets discovered at each level of this
recursion.

This constitutes a divide-and-conquer decomposition: the itemset-mining
problem over the full database is decomposed into one recursive
sub-problem per item, then per pair of items, and so on, each solved on a
structure substantially smaller than the original transaction database.

##  Why Candidate Generation Is Unnecessary

Apriori's search proceeds by generating candidate itemsets and testing
each against the database to determine whether it is frequent — a
generate-then-verify procedure. FP-Growth's recursion instead directly
enumerates the itemsets present in each conditional FP-tree's structure:
a frequent itemset is read off as an existing path in a conditional tree,
rather than proposed and then checked. Because the conditional FP-tree is
built only from the pattern base of an already-frequent item, and because
its own construction inherits the same frequency filtering as Section 2,
every itemset discovered by this recursion is frequent by construction.
This is the precise sense in which FP-Growth requires no separate
candidate-generation step — generation and verification are the same
operation.

##  Complexity Analysis

**Tree construction.** Building the FP-tree requires $O(n \cdot L)$ time,
where $n$ is the number of transactions and $L$ is the average transaction
length, following the initial linear-time counting scan. This is a single
linear pass over the (filtered, reordered) transaction data.

**Mining.** The cost of the recursive mining phase depends on the degree
of compression achieved by the tree. For highly correlated, skewed data,
the conditional trees at each recursive step are small, and mining is
substantially faster than Apriori's candidate-based search. For nearly
uniform or weakly correlated data, the tree achieves little compression,
and FP-Growth's advantage over Apriori diminishes correspondingly.

##  Strengths and Limitations

**Strengths.**
- Requires exactly two scans of the database, independent of the length
  of the longest frequent itemset, in contrast to Apriori's $O(k)$ scans.
- Eliminates candidate generation entirely (Section 4).
- Typically substantially faster than Apriori on real-world transactional
  data, which is usually correlated and skewed in item frequency.

**Limitations.**
- The FP-tree can remain large for datasets that are large, sparse, or
  weakly correlated, in which case little compression is achieved and the
  structural advantage over Apriori is reduced.
- The recursive conditional-tree construction is materially more complex
  to implement correctly than Apriori's level-wise search or Eclat's
  set-intersection approach.
- The algorithm requires the tree, or a sufficient working portion of it,
  to fit in memory.

##  Comparison with Apriori and Eclat

| Property | Apriori | Eclat | FP-Growth |
|---|---|---|---|
| Database scans | $O(k)$, $k$ = longest frequent itemset | 1 (to construct TID-sets) | 2 |
| Data structure | Flat transaction list (horizontal format) | Vertical TID-sets | Compressed prefix tree |
| Candidate generation | Required | Not required (set intersection) | Not required (divide-and-conquer over tree structure) |

All three algorithms rely on the same underlying pruning justification —
the Apriori principle, proved in `Apriori Algorithm/doc/Description.md`,
Section 4 — but differ in how they represent the data and in what
computational mechanism replaces explicit candidate testing.


##  Code Reference

| File | Contents |
|---|---|
| `code/fpgrowth_scratch.py` | FP-tree construction and recursive mining implemented from scratch in pure Python, cross-checked for identical output against the Apriori and Eclat implementations in their respective folders |
| `code/fpgrowth_mlxtend_demo.py` | Equivalent implementation using `mlxtend`'s `fpgrowth` function, with a fallback to the from-scratch version if `mlxtend` is unavailable, and a timing comparison against Apriori |