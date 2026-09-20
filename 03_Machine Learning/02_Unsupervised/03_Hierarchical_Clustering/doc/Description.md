# Hierarchical Clustering



##  Problem Framing

Centroid-based methods such as K-Means require the number of clusters $k$
to be fixed in advance and return a single partition of the data.
Hierarchical clustering instead constructs the entire nested sequence of
partitions, from $n$ singleton clusters to one cluster containing all
points, and represents this sequence as a tree (the dendrogram). The
number of clusters is then a choice made after the hierarchy has been
computed, by selecting a height at which to cut the tree, rather than a
parameter required before computation begins.

Two directions of construction exist:

- **Agglomerative (bottom-up).** Begin with every point as its own
  cluster and repeatedly merge the two closest clusters. This is the
  standard approach and the subject of this document.
- **Divisive (top-down).** Begin with a single cluster containing all
  points and repeatedly split it. This direction is rarely used in
  practice, since it is more computationally expensive and requires
  defining a splitting rule, which is generally harder to specify well
  than a merging rule.

##  Agglomerative Algorithm Specification

**Input:** dataset of $n$ points, a distance metric $d$, a linkage
criterion (Section 3).

1. Initialize $n$ clusters, each containing exactly one point.
2. Compute the pairwise distance between every pair of clusters, using the
   chosen linkage criterion.
3. Identify the two clusters with the smallest linkage distance and merge
   them into a single cluster.
4. Recompute the linkage distance between the newly formed cluster and
   every remaining cluster. Distances between clusters unaffected by the
   merge are unchanged.
5. Repeat steps 3–4 until a single cluster containing all $n$ points
   remains.

**Output.** The sequence of merges, together with the distance at which
each merge occurred, constitutes the dendrogram (Section 4). No
information is discarded: the full hierarchy of partitions, from $n$
clusters down to one, is recoverable from this merge history.

##  Linkage Criteria

The linkage criterion defines the distance between two clusters $A$ and
$B$ as a function of the pairwise distances between their constituent
points, and this choice determines the geometric character of the
resulting clusters.

| Linkage | Definition | Geometric tendency |
|---|---|---|
| Single | $\min_{a \in A,\, b \in B} d(a, b)$ | Favors elongated, "chained" clusters; sensitive to noise bridging two otherwise distinct clusters |
| Complete | $\max_{a \in A,\, b \in B} d(a, b)$ | Favors compact clusters of roughly equal diameter |
| Average | $\dfrac{1}{\lvert A \rvert \lvert B \rvert} \sum_{a \in A} \sum_{b \in B} d(a, b)$ | A compromise between single and complete linkage |
| Ward | Increase in total within-cluster variance resulting from the merge | Favors compact, similarly-sized clusters; the hierarchical analogue of K-Means's variance-minimization objective |

**Chaining under single linkage.** Because single linkage depends only on
the closest pair of points between two clusters, a sequence of points
forming a thin bridge between two otherwise well-separated dense regions
is sufficient to cause them to merge early. This is the formal basis for
single linkage's known sensitivity to noise and its tendency to produce
elongated clusters, and it is a direct consequence of the linkage
definition rather than an implementation artifact.

**Correspondence between Ward linkage and K-Means.** Ward's criterion
merges the pair of clusters that produces the smallest increase in total
within-cluster sum of squared distances — the same quantity K-Means
minimizes directly (see `K-Means/doc/Description.md`). Ward linkage can
therefore be understood as applying K-Means's objective greedily and
hierarchically, rather than through iterative centroid reassignment.

##  The Dendrogram as a Formal Object

A dendrogram is a binary tree in which each leaf corresponds to a single
data point, each internal node corresponds to a merge event recorded in
Step 3 of the algorithm, and the height of an internal node equals the
linkage distance at which that merge occurred.

**Cutting the dendrogram.** A horizontal cut at height $h$ intersects the
tree at the set of clusters that existed immediately before any merge with
linkage distance exceeding $h$. The number of clusters obtained equals the
number of vertical lines crossed by the cut. Choosing $h$ is therefore
equivalent to choosing the number of output clusters, and this choice can
be deferred until after the full hierarchy has been computed.

**Heuristic for choosing a cut height.** A large gap between consecutive
merge heights — visible as an unusually long vertical segment in the
dendrogram with no nearby merges — indicates that the two structures being
separated at that height are more dissimilar than typical merges at
neighboring heights, and is commonly used as a heuristic signal for a
natural number of clusters.

##  Complexity Analysis

Naive agglomerative clustering requires $O(n^3)$ time and $O(n^2)$ space:
the $O(n^2)$ space is required to store the full pairwise distance matrix,
and the $O(n^3)$ time follows from performing $n-1$ merges, each requiring
an $O(n^2)$ search over the current distance matrix to identify the
closest pair. This is substantially more expensive than K-Means's $O(nk)$
per-iteration cost and is the primary factor limiting hierarchical
clustering's scalability. Optimized implementations, such as those based
on nearest-neighbor chains, reduce this to $O(n^2 \log n)$ or $O(n^2)$
depending on the linkage criterion, but the $O(n^2)$ space requirement
remains regardless of the time-complexity optimization used.

##  Strengths and Limitations

**Strengths.**
- Does not require the number of clusters to be fixed before computation;
  the choice is deferred to dendrogram interpretation (Section 4).
- Deterministic: unlike K-Means, the result does not depend on a random
  initialization, so no equivalent of multiple restarts is required.
- Recovers nested, multi-scale structure directly, since the full
  hierarchy — not a single partition — is the output.
- Applicable with any valid distance metric, not only Euclidean distance,
  which broadens its applicability beyond centroid-based methods that
  require averaging in the original feature space.

**Limitations.**
- The $O(n^2)$ or worse time and space complexity (Section 5) limits
  scalability to large datasets.
- Merges are greedy and irrevocable: once two clusters are merged, no
  later step can separate them, so an early suboptimal merge propagates
  through every subsequent level of the hierarchy and cannot be corrected.
- Sensitivity to the choice of linkage criterion and to noise or outliers
  is substantial, particularly under single linkage (Section 3).


##  Code Reference

| File | Contents |
|---|---|
| `code/hierarchical_scratch.py` | Agglomerative clustering implemented from scratch in NumPy, supporting single, complete, average, and Ward linkage; constructs the merge history required to draw a dendrogram, with a `cut_tree(k)` helper |
| `code/hierarchical_sklearn_demo.py` | Equivalent implementation using `scipy.cluster.hierarchy` for dendrogram plotting and `sklearn.cluster.AgglomerativeClustering` |