# DBSCAN



##  Problem Framing

Centroid-based methods such as K-Means define a cluster by a center point
and assign every observation to the nearest such center, which forces
every point — including outliers — into some cluster. DBSCAN instead
defines a cluster as a **maximal connected region of high point density**,
separated from other such regions by areas of low density. This framing
has two direct consequences: clusters may take arbitrary shape rather than
being constrained to convex regions around a centroid, and points that do
not belong to any dense region are left unassigned rather than forced into
the nearest one.

##  Formal Definitions

Let $\varepsilon$ (eps) be a neighborhood radius and let `minPts` be a
minimum neighborhood size, both fixed in advance.

| Term | Definition |
|---|---|
| $\varepsilon$-neighborhood of $p$ | $N_\varepsilon(p) = \{q \in D : d(p, q) \leq \varepsilon\}$ |
| Core point | $p$ such that $\lvert N_\varepsilon(p) \rvert \geq \text{minPts}$ |
| Border point | $p$ with $\lvert N_\varepsilon(p) \rvert < \text{minPts}$, but $p \in N_\varepsilon(q)$ for some core point $q$ |
| Noise point | A point that is neither core nor border |
| Directly density-reachable | $q$ is directly density-reachable from $p$ if $p$ is a core point and $q \in N_\varepsilon(p)$ |
| Density-reachable | $q$ is density-reachable from $p$ if there exists a chain $p = p_1, p_2, \dots, p_n = q$ such that $p_{i+1}$ is directly density-reachable from $p_i$ for each $i$ |
| Density-connected | $p$ and $q$ are density-connected if there exists a point $o$ from which both $p$ and $q$ are density-reachable |

**Cluster definition.** A cluster $C$ is a non-empty subset of $D$
satisfying two properties: (i) maximality — if $p \in C$ and $q$ is
density-reachable from $p$, then $q \in C$; and (ii) connectivity — every
pair of points in $C$ is density-connected. Under this definition, a
cluster is precisely a maximal set of density-connected points, and the
set of all clusters together with the noise points partitions $D$.

##  Algorithm Specification

**Input:** dataset $D$, parameters $\varepsilon$ and `minPts`.

1. Select an unvisited point $p \in D$ and mark it visited.
2. Compute $N_\varepsilon(p)$.
3. If $\lvert N_\varepsilon(p) \rvert \geq \text{minPts}$: $p$ is a core
   point. Initialize a new cluster containing $p$, then expand it by
   iteratively adding every point density-reachable from $p$ — that is,
   for each core point encountered during expansion, add its
   $\varepsilon$-neighborhood to the cluster and continue the search from
   any newly added core points.
4. If $\lvert N_\varepsilon(p) \rvert < \text{minPts}$: provisionally label
   $p$ as noise. This label is not final, since $p$ may later be found
   within the $\varepsilon$-neighborhood of a core point discovered in a
   subsequent iteration, at which point it is reclassified as a border
   point of that cluster.
5. Repeat steps 1–4 until every point in $D$ has been visited.

**Output:** a partition of $D$ into clusters, together with a residual set
of points labeled noise.

##  Complexity Analysis

The dominant cost is the repeated neighborhood query $N_\varepsilon(p)$,
executed once per point.

| Implementation | Per-query cost | Total complexity |
|---|---|---|
| Naive (pairwise distance) | $O(n)$ | $O(n^2)$ |
| Spatial index (KD-tree or Ball-tree) | $O(\log n)$, effective in low-to-moderate dimensions | $O(n \log n)$ |

The spatial-index speedup degrades as dimensionality increases, since
tree-based neighbor search loses its advantage over linear scan once the
curse of dimensionality reduces the discriminative power of distance


##  Parameter Selection

**`minPts`.** A standard heuristic is $\text{minPts} \geq D + 1$, where $D$
is the number of feature dimensions; $\text{minPts} = 2D$ is a common
default, and small fixed values (4–10) are typical for two-dimensional
data. Increasing `minPts` makes the algorithm more robust to noise but
increases the risk of merging distinct clusters that are connected by a
thin bridge of points.

**$\varepsilon$.** The standard method is the **k-distance plot**: for
every point, compute the distance to its $k$-th nearest neighbor, with
$k = \text{minPts}$, and sort these values in ascending order. The
resulting curve typically has a "knee" — a point of sharp increase in
slope — and the distance at this knee is used as $\varepsilon$. The
justification is that points in dense regions have small $k$-distance
values, while the knee marks the transition to points in sparse regions or
noise.

## 6. Strengths and Limitations

**Strengths.**
- Recovers clusters of arbitrary shape, including non-convex structures
  such as moons, rings, and nested shapes, which violate K-Means's
  isotropic-cluster assumption.
- Determines the number of clusters automatically as a byproduct of the
  density-reachability structure, rather than requiring it as an input.
- Produces an explicit noise label, which is directly usable as an
  anomaly-detection signal (see `Anomaly Detection with Clustering`).
- Robust to outliers in the sense that outliers are excluded rather than
  distorting a cluster representative, unlike centroid-based methods where
  a single outlier can shift a centroid.

**Limitations.**
- A single global $\varepsilon$ cannot simultaneously be appropriate for
  clusters of substantially different density: a value small enough to
  resolve a dense cluster will fragment or eliminate a sparser one, and a
  value large enough to preserve the sparse cluster will merge it with
  denser neighboring structure.
- Performance is sensitive to the choice of $\varepsilon$ and `minPts`, and
  degrades in high dimensions, where the curse of dimensionality causes
  pairwise distances to concentrate (the empty-space phenomenon), reducing
  the discriminative power of the $\varepsilon$-neighborhood criterion.
- The algorithm is not fully deterministic at cluster boundaries: a border
  point reachable from two distinct clusters is assigned to whichever
  cluster's core point is processed first, so the final assignment can
  depend on traversal order.

The density-uniformity limitation is addressed directly by **HDBSCAN**
(Hierarchical DBSCAN), which builds a hierarchy over a range of density
thresholds rather than fixing a single global $\varepsilon$, and extracts
the most stable clusters across that hierarchy. This allows it to handle
variable-density data that defeats standard DBSCAN.


##  Code Reference

| File | Contents |
|---|---|
| `code/dbscan_scratch.py` | NumPy implementation of DBSCAN from first principles: region query, core/border/noise labeling, and cluster expansion via breadth-first search |
| `code/dbscan_sklearn_demo.py` | `sklearn.cluster.DBSCAN` with a k-distance plot for $\varepsilon$ selection, and a comparison against K-Means on the two-moons dataset illustrating DBSCAN's arbitrary-shape advantage |