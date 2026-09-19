# FP-Growth Algorithm (Frequent Pattern Growth)

## 1. Motivation — fixing Apriori's biggest weakness
Apriori's main cost is **repeated full-database scans** — one scan per
itemset-size level (see `Apriori Algorithm/doc/Description.md`). FP-Growth mines
frequent itemsets with only **two scans of the database, total**, by
compressing the transactions into a compact tree structure (the
**FP-tree**) and then mining that tree directly — no candidate generation
at all.

## 2. The FP-tree — a compressed representation of all transactions
1. **Scan 1**: count the support of every individual item.
2. Discard infrequent items; sort the remaining items in each transaction
   by **descending global frequency** (this ordering is the key trick — it
   maximizes how much different transactions share common prefixes).
3. **Scan 2**: insert each (reordered, filtered) transaction into a shared
   prefix tree:
   - Start at the tree's root.
   - For each item in the transaction (in frequency order), if a child
     with that item already exists, increment its count; otherwise create
     a new child node with count 1.
   - Follow/extend the path down the tree for the rest of the transaction.
4. Maintain a **header table**: for each item, a pointer to its *first*
   occurrence in the tree, and each tree node also links to the *next* node
   with the same item ("node-links") — this lets you quickly find every
   place a given item appears in the tree without rescanning transactions.

Because transactions sharing a frequent prefix (e.g. many baskets starting
with "bread, milk, ...") reuse the same tree path, the FP-tree is often
**much smaller than the raw transaction list** for real, correlated retail
data.

## 3. Mining the FP-tree — the "growth" part
For each item `i` (processed from least to most frequent, to keep
sub-problems small), build its **conditional pattern base**: the set of
all paths in the tree that lead to a node for `i` (i.e., all the "prefix
paths" of transactions that contain `i`), extracted by following `i`'s
node-links from the header table.

From this conditional pattern base, build a smaller **conditional FP-tree**
(same construction process, recursively) — this tree represents "all
itemsets that co-occur with `i`." Recursively mine *that* smaller tree the
same way. Every frequent itemset containing `i` is found by combining `i`
with whatever's frequent in its conditional FP-tree, recursively, all the
way down to single-item conditional trees.

This is a **divide-and-conquer** strategy: the frequent-itemset-mining
problem for the whole database is broken into many much smaller
sub-problems (one recursive conditional tree per item, then per pair of
items, etc.), each solved on a much smaller structure than the original
database.

## 4. Why no candidate generation is needed
Apriori has to *generate* candidate itemsets and then check if they're
frequent. FP-Growth instead directly *reads off* the frequent itemsets that
already exist as tree paths in each conditional FP-tree — the itemsets
"grow" out of the tree structure itself (hence the name), so there's no
guess-and-check candidate step at all.

## 5. Complexity
- Building the FP-tree: `O(n * L)` (n transactions, L = average
  transaction length) — a single linear pass (after the initial counting
  scan).
- Mining: depends on the tree's structure/compression — in the best case
  (very compressible, correlated data), dramatically faster than Apriori;
  in the worst case (nearly uniform/random data, poor compression),
  advantages shrink.

## 6. Strengths / Weaknesses
**Strengths**: only 2 database scans (vs. Apriori's k scans, one per
itemset-size level); no candidate generation; often much faster on
real-world (correlated, skewed-frequency) transactional data.
**Weaknesses**: the FP-tree can still become large and complex for very
large, sparse, or low-correlation datasets (little compression benefit);
implementation is notably more complex than Apriori or Eclat (recursive
tree construction/mining is fiddly to get right); needs the whole tree (or
enough of it) to fit reasonably in memory.

## 7. Relation to Apriori and Eclat
| | Data scans | Structure | Candidate generation? |
|---|---|---|---|
| Apriori | O(k) (k = longest itemset) | flat list of transactions | Yes |
| Eclat | 1 (to build TID-sets) | vertical TID-sets | No (set intersection) |
| FP-Growth | 2 | compressed prefix tree | No (divide & conquer on tree) |

## 8. Basic → Pro
1. **Basic**: build the FP-tree for a small transaction set by hand/code,
   inspect the header table and node-links.
2. **Intermediate**: recursive conditional-pattern-base extraction and
   conditional FP-tree mining for 2- and 3-itemsets.
3. **Pro**: benchmark against Apriori/Eclat on a larger, more realistic
   (skewed-frequency) synthetic transaction dataset to see FP-Growth's
   scan-count advantage translate into real speedup.

## 9. Code in this folder
- `code/fpgrowth_scratch.py` — FP-tree construction and recursive mining
  from scratch in pure Python, cross-checked for identical output against
  the Apriori and Eclat implementations in their own folders.
- `code/fpgrowth_mlxtend_demo.py` — the same problem via `mlxtend`'s
  `fpgrowth` (with a fallback to the from-scratch version if `mlxtend`
  isn't installed), plus a rough timing comparison against Apriori.
