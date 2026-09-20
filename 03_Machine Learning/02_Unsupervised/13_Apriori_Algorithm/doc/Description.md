# Apriori Algorithm


##  Problem Definition

**Association rule mining** identifies statistically significant
co-occurrence relationships between items in a transactional database. The
canonical application is market basket analysis: determining that
customers who purchase bread and butter also tend to purchase milk.

The task is unsupervised. There is no target variable to predict; the
objective is to discover co-occurrence structure directly from the data.

##  Formal Preliminaries

Let $I = \{i_1, i_2, \dots, i_m\}$ be the set of all distinct items, and let
a transaction database $D$ consist of transactions $T_1, \dots, T_n$, each
a subset of $I$.

| Term | Definition |
|---|---|
| Itemset | Any subset $X \subseteq I$, e.g. $\{\text{bread}, \text{butter}\}$ |
| Support | $\text{support}(X) = \dfrac{\lvert \{T \in D : X \subseteq T\} \rvert}{\lvert D \rvert}$ |
| Frequent itemset | An itemset $X$ with $\text{support}(X) \geq \text{min\_support}$ |
| Association rule | $X \rightarrow Y$, with $X, Y \subseteq I$ disjoint |
| Confidence | $\text{confidence}(X \rightarrow Y) = \dfrac{\text{support}(X \cup Y)}{\text{support}(X)}$ |
| Lift | $\text{lift}(X \rightarrow Y) = \dfrac{\text{confidence}(X \rightarrow Y)}{\text{support}(Y)}$ |

**Interpretation of lift.** Lift measures deviation from statistical
independence between $X$ and $Y$:

- $\text{lift} > 1$: positive association — $Y$ occurs more often given $X$
  than its baseline frequency would predict.
- $\text{lift} \approx 1$: $X$ and $Y$ are approximately independent.
- $\text{lift} < 1$: negative association — the presence of $X$ makes $Y$
  less likely than baseline.

Confidence alone is insufficient to establish a meaningful association,
since a rule can have high confidence purely because $Y$ is frequent
overall; lift corrects for this by normalizing against $\text{support}(Y)$.

##  The Combinatorial Obstacle

For $m$ distinct items, the number of possible non-empty itemsets is
$2^m - 1$. Exhaustively computing the support of every itemset against
every transaction is infeasible for any realistic catalog size — for
$m = 100$ items, $2^{100} - 1$ exceeds $10^{30}$ candidate itemsets. The
Apriori algorithm's central contribution is a principled method for
eliminating the overwhelming majority of these candidates without
computing their support directly.

##  The Apriori Principle

**Statement.** If an itemset $X$ is frequent, every subset $X' \subseteq X$
is also frequent. Equivalently, by contraposition: if an itemset is
infrequent, no superset of it can be frequent.

**Proof.** For $X' \subseteq X$, every transaction containing $X$ must
also contain $X'$, since $X' \subseteq X \subseteq T$. Therefore
$\{T : X \subseteq T\} \subseteq \{T : X' \subseteq T\}$, which implies
$\text{support}(X') \geq \text{support}(X)$. If $X$ is frequent, i.e.
$\text{support}(X) \geq \text{min\_support}$, it follows that
$\text{support}(X') \geq \text{min\_support}$ as well. $\blacksquare$

**Consequence.** This monotonicity property is what makes level-wise
pruning valid: a $(k+1)$-itemset need only be considered as a candidate if
all of its $k$-subsets are already known to be frequent. This eliminates
the need to compute the support of any itemset with an infrequent subset,
which in practice removes the large majority of the $2^m - 1$ candidates
identified in Section 3.

##  Algorithm Specification

**Input:** transaction database $D$, thresholds $\text{min\_support}$ and
$\text{min\_confidence}$.

1. **Pass 1.** Scan $D$ and compute the support of every individual item
   (1-itemset). Retain those meeting $\text{min\_support}$ as $L_1$.
2. **Candidate generation ($k = 2$).** Generate candidate 2-itemsets by
   pairing items from $L_1$ only. By the Apriori principle, any 2-itemset
   containing an item absent from $L_1$ cannot be frequent and is excluded
   without being evaluated.
3. **Pass 2.** Scan $D$ and compute the support of each candidate
   2-itemset. Retain those meeting $\text{min\_support}$ as $L_2$.
4. **Iteration.** For $k = 3, 4, \dots$: generate candidate $(k+1)$-itemsets
   by joining pairs of frequent $k$-itemsets in $L_k$ that share $k-1$
   items, then prune any candidate for which some $k$-subset is not in
   $L_k$ (the join-and-prune step). Scan $D$ to compute support for the
   surviving candidates and retain those meeting $\text{min\_support}$ as
   $L_{k+1}$.
5. **Termination.** Stop when $L_{k+1} = \emptyset$.
6. **Rule generation.** For every frequent itemset $Z = X \cup Y$ found in
   steps 1–5, and for every way of partitioning $Z$ into disjoint,
   non-empty $X$ and $Y$, form the rule $X \rightarrow Y$ and retain it if
   $\text{confidence}(X \rightarrow Y) \geq \text{min\_confidence}$.

##  Complexity Analysis

**Database scans.** The number of full scans of $D$ equals the length of
the longest frequent itemset, since each level $k$ requires one pass to
evaluate the candidates generated from $L_{k-1}$. This repeated,
full-database scanning is Apriori's dominant cost and its primary
bottleneck. FP-Growth (see `FP-Growth Algorithm/doc/Description.md`)
addresses this directly by compressing the database into a single
in-memory structure after exactly two scans, regardless of itemset length.

**Candidate generation.** In the worst case, the number of candidates
generated at each level remains exponential in $m$. In practice, because
real transactional data is typically sparse, the Apriori-principle pruning
in Section 4 keeps the candidate set tractable; the algorithm's practical
performance depends heavily on how low $\text{min\_support}$ is set
relative to the data's density.

## Strengths and Limitations

**Strengths.**
- Conceptually simple and easy to implement correctly.
- Exact: returns all frequent itemsets above threshold, not an
  approximation.
- The Apriori-principle pruning (Section 4) is a foundational technique
  reused, in modified form, by later algorithms in this space.

**Limitations.**
- I/O cost from repeated full-database scans (Section 6).
- Candidate generation can still be large when $\text{min\_support}$ is low
  or the data is dense.
- Largely superseded in practice by **FP-Growth**, which eliminates
  candidate generation entirely, and **Eclat**, which uses a depth-first
  search over transaction-id sets. Both are covered in their respective
  repository folders.


##  Code Reference

| File | Contents |
|---|---|
| `code/apriori_scratch.py` | Pure-Python implementation of level-wise frequent itemset mining with Apriori-principle pruning, and association rule generation with support, confidence, and lift |
| `code/apriori_mlxtend_demo.py` | Equivalent implementation using `mlxtend`'s `apriori` and `association_rules` functions, with a pandas-only fallback if `mlxtend` is unavailable |