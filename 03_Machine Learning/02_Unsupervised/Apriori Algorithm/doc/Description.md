# Apriori Algorithm (Association Rule Mining)

## 1. What problem does this solve?
**Association rule mining** finds relationships between items that
co-occur together in transactions — the classic example is **market basket
analysis**: "customers who buy bread and butter also tend to buy milk."
This is unsupervised: there's no label to predict, we're discovering
patterns of co-occurrence directly from the data.

## 2. Key definitions
Given a database of transactions (each transaction = a *set* of items):
- **Itemset**: any set of items, e.g. `{bread, butter}`.
- **Support**: `support(X) = (# transactions containing X) / (total # transactions)`
  — how frequently itemset `X` appears.
- **Frequent itemset**: an itemset whose support ≥ a chosen `min_support`
  threshold.
- **Association rule**: `X -> Y` (if a transaction contains itemset `X`,
  it's likely to also contain itemset `Y`), where `X` and `Y` are disjoint
  itemsets.
- **Confidence**: `confidence(X -> Y) = support(X ∪ Y) / support(X)` —
  among transactions containing `X`, what fraction also contain `Y`?
- **Lift**: `lift(X -> Y) = confidence(X -> Y) / support(Y)` — how much
  more likely is `Y` given `X`, compared to `Y`'s baseline frequency?
  `lift > 1` means a positive association (not just coincidental
  co-occurrence); `lift ≈ 1` means `X` and `Y` are roughly independent;
  `lift < 1` means a negative association.

## 3. The core challenge: combinatorial explosion
With `m` distinct items, there are `2^m - 1` possible non-empty itemsets —
checking every single one against every transaction is completely
infeasible even for a moderate catalog size. Apriori's entire contribution
is a smart way to avoid checking most of them.

## 4. The Apriori Principle (key insight)
> **If an itemset is frequent, then all of its subsets must also be
> frequent.** Equivalently (contrapositive): **if an itemset is
> infrequent, none of its supersets can be frequent.**

This is intuitively obvious (adding more required items to a set can only
keep support the same or make it lower, never higher) but it's extremely
powerful: it lets us **prune** the search space aggressively.

## 5. Algorithm
1. **Pass 1**: scan the database, count support for every individual item
   (1-itemsets). Keep only those with support ≥ `min_support` → `L1`.
2. **Candidate generation**: generate candidate 2-itemsets by combining
   pairs of items from `L1`. (Only from *frequent* 1-itemsets — this is the
   pruning step, using the Apriori principle: any 2-itemset containing an
   infrequent item can't be frequent.)
3. **Pass 2**: scan the database again, count support for each 2-itemset
   candidate. Keep frequent ones → `L2`.
4. **Repeat**: generate candidate `(k+1)`-itemsets from `Lk` (only combining
   frequent k-itemsets whose union has size k+1 and whose every k-subset is
   already known frequent — the "join and prune" step), scan the database,
   filter by `min_support` → `L(k+1)`.
5. Stop when no new frequent itemsets are found.
6. **Generate rules**: for every frequent itemset, split it into `X -> Y`
   in all possible ways and keep rules meeting a `min_confidence` threshold.

## 6. Complexity
- The number of database scans equals the length of the longest frequent
  itemset — this **repeated full-database scanning** is Apriori's main
  bottleneck (see `FP-Growth Algorithm/doc/Description.md` for a method that
  scans the database only twice, total, regardless of itemset length).
- Candidate generation itself can still be expensive when there are many
  frequent itemsets at each level (still exponential in the worst case),
  but pruning via the Apriori principle keeps it tractable in practice for
  real transactional data (which is usually sparse).

## 7. Strengths / Weaknesses
**Strengths**: simple, intuitive, exact (finds *all* frequent itemsets
above the threshold, not an approximation); the Apriori-principle pruning
is a classic, elegant idea.
**Weaknesses**: many repeated full-database scans (I/O-heavy); can still
generate a huge number of candidates when `min_support` is low or the data
is dense; superseded in practice by **FP-Growth** (no candidate
generation) and **Eclat** (depth-first, uses transaction-id sets) for
larger datasets.

## 8. Basic → Pro
1. **Basic**: 1-itemset counting, `min_support` filtering.
2. **Intermediate**: full level-wise candidate generation + Apriori-
   principle pruning, up to k-itemsets.
3. **Pro**: rule generation with confidence + lift filtering; compare
   runtime/candidate-count against `Eclat`/`FP-Growth` on the same data
   (see those folders) to see why Apriori is rarely used at scale today.

## 9. Code in this folder
- `code/apriori_scratch.py` — Apriori from scratch in pure Python: frequent
  itemset mining (level-wise, with Apriori-principle pruning) and
  association rule generation with support/confidence/lift.
- `code/apriori_mlxtend_demo.py` — the same problem using the widely-used
  `mlxtend` library's `apriori` + `association_rules` (falls back to a
  small pandas-only implementation if `mlxtend` isn't installed).
