# Eclat Algorithm



##  Problem Correspondence with Apriori

Eclat solves the identical problem addressed by Apriori — finding all
itemsets whose support meets a threshold `min_support`, as a basis for
association rule mining (see `Apriori Algorithm/doc/Description.md` for
the shared definitions of support, confidence, and lift). The two
algorithms differ along two independent axes:

| Axis | Apriori | Eclat |
|---|---|---|
| Search order | Breadth-first (level-wise: all frequent $k$-itemsets before any $(k+1)$-itemset) | Depth-first |
| Data representation | Horizontal — a list of transactions, each a set of items | Vertical — for each item, the set of transaction identifiers containing it |

Both algorithms rely on the same underlying pruning result, the Apriori
principle (proved in `Apriori Algorithm/doc/Description.md`, Section 4):
if an itemset is infrequent, no superset of it can be frequent. Eclat
differs from Apriori only in how it computes support and in what order it
explores the search space; the correctness argument for pruning is
identical in both algorithms.

##  The Vertical Data Format

**Horizontal format** (used by Apriori) represents the database as a list
of transactions:

```
T1: {bread, milk}
T2: {bread, diaper, beer}
```

**Vertical format** (used by Eclat) inverts this representation, storing
for each item the set of transaction identifiers (the TID-set) in which it
appears:

```
bread:  {T1, T2}
milk:   {T1}
diaper: {T2}
beer:   {T2}
```

**Support as set intersection.** Under the vertical format, the support of
a union of items reduces to a set-cardinality computation. For itemset $X$
with TID-set $t(X)$, and an item $i$ with TID-set $t(\{i\})$:

$$t(X \cup \{i\}) = t(X) \cap t(\{i\}), \qquad \text{support}(X \cup \{i\}) = \frac{\lvert t(X) \cap t(\{i\}) \rvert}{n}$$

This holds because a transaction contains $X \cup \{i\}$ if and only if it
contains both $X$ and $i$, i.e., if and only if its identifier appears in
both TID-sets. Consequently, computing the support of any candidate
itemset requires no scan of the transaction database once the base
TID-sets have been constructed — it requires only an intersection of two
sets already held in memory.

##  Algorithm Specification

**Input:** transaction database $D$ in vertical format, threshold
`min_support`.

1. Construct the TID-set $t(\{i\})$ for every individual item $i$, by a
   single pass over $D$. Retain items whose support meets `min_support` as
   the frequent 1-itemsets, using the same pruning justification as
   Apriori: only itemsets extending an already-frequent itemset can
   themselves be frequent.
2. Fix a total order on items (e.g., lexicographic). For each frequent
   itemset $X$ found so far, and for each frequent item $i$ that succeeds
   every item already in $X$ under this order, compute the candidate
   itemset $X \cup \{i\}$ and its TID-set via Section 2's intersection
   identity. Fixing an order in this way ensures each itemset is generated
   exactly once, rather than once per permutation of its elements.
3. If $\lvert t(X \cup \{i\}) \rvert / n \geq \text{min\_support}$, the
   candidate is frequent; recurse on it depth-first, extending it further
   under the same procedure.
4. If the candidate does not meet `min_support`, do not recurse into it,
   by the Apriori principle: no extension of an infrequent itemset can be
   frequent.
5. The search terminates along each branch when no further frequent
   extension exists; the union of all itemsets discovered across all
   branches is the complete set of frequent itemsets.

This procedure explores the itemset lattice as a tree, descending only
into branches rooted at itemsets already confirmed frequent. No separate
candidate-generation pass, as used in Apriori's join step, is required:
generation and support computation are unified into a single recursive
step.

##  Basis for Performance Improvement over Apriori

Three properties of the vertical-format, depth-first approach account for
Eclat's typical performance advantage over Apriori:

1. **Elimination of repeated database scans.** After the single initial
   pass used to construct the base TID-sets (Step 1), every subsequent
   support computation is a set intersection performed entirely in memory.
   This removes the dependency, present in Apriori, between the number of
   database scans and the length of the longest frequent itemset (see
   `Apriori Algorithm/doc/Description.md`, Section 6).
2. **Lower peak memory footprint.** Depth-first search maintains only the
   itemsets along the current search path, rather than materializing every
   candidate at a given level simultaneously, as Apriori's level-wise
   breadth-first search does. This is a structural property of depth-first
   versus breadth-first traversal and is independent of the data
   representation.
3. **Favorable scaling with sparsity.** When each item appears in
   relatively few transactions, TID-sets remain small, which keeps
   intersection operations inexpensive. This makes the method
   particularly effective on sparse transactional data, which is typical
   of real-world market-basket datasets.

##  Complexity Analysis

In the best case, using efficient set or bit-vector intersection, Eclat's
practical performance on sparse transactional data substantially exceeds
Apriori's. This is a practical, not asymptotic, distinction: the
worst-case complexity of itemset search remains combinatorial in both
algorithms, governed by the same $2^m - 1$ candidate space established in
`Apriori Algorithm/doc/Description.md`, Section 3. The Apriori-principle
pruning that keeps Eclat tractable in practice is the same pruning
argument that keeps Apriori tractable; the two algorithms share their
worst-case complexity class and differ only in constant-factor and
memory-access-pattern performance.

##  Strengths and Limitations

**Strengths.**
- No explicit candidate-generation step; candidate construction and
  support evaluation are unified (Section 3).
- Support computation via set intersection is fast, particularly with a
  bit-vector representation of TID-sets, where intersection reduces to a
  bitwise AND operation.
- Depth-first search is memory-efficient relative to Apriori's
  level-by-level candidate storage.
- Particularly effective on sparse data, where small TID-sets keep
  intersection costs low.

**Limitations.**
- TID-sets can become large for items with high frequency or for very
  large transaction databases, increasing memory consumption.
- Eclat does not provide a data structure as compact as FP-Growth's
  FP-tree for very dense data (see
  `FP-Growth Algorithm/doc/Description.md`), and can be less efficient
  than FP-Growth in that regime.

##  Code Reference

| File | Contents |
|---|---|
| `code/eclat_scratch.py` | Pure-Python implementation of Eclat: vertical TID-set construction, depth-first frequent itemset mining via set intersection, and a runtime comparison against the Apriori implementation in `Apriori Algorithm/code/apriori_scratch.py` on identical transaction data |