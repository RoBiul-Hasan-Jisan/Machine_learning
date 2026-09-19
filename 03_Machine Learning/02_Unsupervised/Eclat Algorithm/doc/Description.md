# Eclat Algorithm (Equivalence Class Transformation)

## 1. Intuition
Eclat solves the exact same problem as Apriori (finding frequent itemsets
for association rule mining — see `Apriori Algorithm/doc/Description.md` for the
shared vocabulary: support, confidence, lift), but takes a fundamentally
different computational approach:

- **Apriori** is *breadth-first* (level-wise: find all frequent 1-itemsets,
  then all frequent 2-itemsets, ...) and is *horizontal* (represents data
  as a list of transactions, each a set of items) — computing support
  means scanning the whole transaction list for each candidate.
- **Eclat** is *depth-first* and *vertical* — for each item, it stores the
  **set of transaction IDs (TID-set)** that contain it. Support becomes a
  simple **set intersection**: `support(X ∪ {new_item}) count = |TIDset(X) ∩ TIDset(new_item)|`.

## 2. The vertical data format
Instead of:
```
T1: {bread, milk}
T2: {bread, diaper, beer}
```
Eclat represents the same data as, for each item, the list of transactions
containing it:
```
bread:  {T1, T2}
milk:   {T1}
diaper: {T2}
beer:   {T2}
```
Now, "how many transactions contain both bread and milk?" is just
`|{T1,T2} ∩ {T1}| = |{T1}| = 1` — a single set intersection, no scanning
required.

## 3. Algorithm (depth-first search)
1. Compute the TID-set for every single item; keep those meeting
   `min_support` (same Apriori-principle pruning applies: any frequent
   itemset's subsets must be frequent too, so we only ever extend
   already-frequent itemsets).
2. For each frequent itemset `X` (starting from single items), and for
   each other frequent item `i` that comes after `X`'s items in some fixed
   ordering (to avoid re-generating the same itemset twice), compute the
   candidate `X ∪ {i}`'s TID-set as `TIDset(X) ∩ TIDset({i})`.
3. If `|TIDset(X ∪ {i})| / n >= min_support`, it's frequent — recurse
   (extend it further) depth-first.
4. This naturally explores the search space as a tree, only ever
   descending into branches built from itemsets already known to be
   frequent — no separate "candidate generation" pass like Apriori.

## 4. Why is this often faster than Apriori?
- **No repeated full-database scans** — after the initial pass to build
  TID-sets, all further support computations are pure set intersections
  (fast, especially with bitset representations).
- **Depth-first** exploration uses less memory at any given moment than
  Apriori's breadth-first level-by-level candidate storage (though the
  TID-sets themselves can be large for very frequent items).
- Particularly effective when the data is **sparse** (each item appears in
  relatively few transactions) — the TID-sets stay small, so intersections
  are cheap.

## 5. Complexity
- Best case, using efficient set/bitset intersections: much faster in
  practice than Apriori on sparse transactional data, though worst-case
  itemset-search complexity is still combinatorial (same fundamental
  problem — pruning via the Apriori principle keeps it tractable, exactly
  as with Apriori).

## 6. Strengths / Weaknesses
**Strengths**: no candidate-generation step, fast support computation via
set intersection, memory-efficient depth-first search, especially strong
on sparse data.
**Weaknesses**: TID-sets can become large (memory-heavy) for very
frequent/dense items or very large transaction databases; doesn't
directly give you an ordering/counting structure as compact as FP-Growth's
FP-tree for extremely dense data (see `FP-Growth Algorithm/doc/Description.md`).

## 7. Basic → Pro
1. **Basic**: vertical TID-set representation, set-intersection-based
   support counting for single items and pairs.
2. **Intermediate**: full depth-first recursive itemset mining with
   Apriori-principle pruning built directly into the recursion.
3. **Pro**: bitset (bit-vector) TID-set representation instead of Python
   sets — dramatically faster intersections (bitwise AND) for larger
   databases; compare wall-clock time against `Apriori Algorithm/` and
   `FP-Growth Algorithm/` on the same dataset.

## 8. Code in this folder
- `code/eclat_scratch.py` — Eclat from scratch in pure Python: vertical
  TID-set construction, depth-first frequent itemset mining via set
  intersection, and a runtime comparison against the Apriori
  implementation from `Apriori Algorithm/code/apriori_scratch.py` on the
  same transactions.
