# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## 1. Intuition
DBSCAN defines clusters as **dense regions of points separated by sparse
regions**. Instead of centroids, it grows clusters outward from dense
neighborhoods. Points that don't belong to any dense region are labeled
**noise/outliers** — a huge advantage over K-Means, which is forced to assign
*every* point to some cluster.

## 2. Key definitions
- **ε (eps)**: radius of the neighborhood around a point.
- **minPts**: minimum number of points (including itself) required within
  that ε-radius for a point to be considered "dense" (a **core point**).
- **Core point**: has ≥ minPts points within ε of it.
- **Border point**: has < minPts neighbors, but lies within ε of a core point.
- **Noise point**: neither core nor border — doesn't belong to any cluster.
- **Density-reachable**: point `q` is density-reachable from `p` if there's a
  chain of core points connecting them, each within ε of the next.

## 3. Algorithm
1. Pick an unvisited point `p`.
2. Find all points within distance ε of `p` (its ε-neighborhood).
3. If `|neighborhood| >= minPts`: `p` is a core point → start a new cluster,
   and recursively add every point density-reachable from `p` (expanding
   through other core points found along the way).
4. If `|neighborhood| < minPts`: mark `p` as noise *for now* (it might later
   be picked up as a border point of some other cluster).
5. Repeat for the next unvisited point until all points are visited.

## 4. Complexity
- Naive: `O(n^2)` (check every pair's distance).
- With a spatial index (KD-tree / Ball-tree, good for low-medium
  dimensions): `O(n log n)`.

## 5. Choosing ε and minPts
- **minPts**: rule of thumb `minPts >= D + 1` (D = number of dimensions);
  commonly `minPts = 2*D` or simply a fixed value like 4–10 for 2D data.
  Larger minPts → more robust to noise but risks merging real clusters.
- **ε**: use a **k-distance plot** — for each point compute the distance to
  its k-th nearest neighbor (k = minPts), sort these distances ascending, and
  look for the "knee" (sharp bend) in the curve. That distance is a good ε.

## 6. Strengths / Weaknesses
**Strengths**:
- Finds clusters of **arbitrary shape** (not just convex blobs) — handles
  moons, rings, nested shapes that break K-Means.
- Automatically determines the number of clusters.
- Explicitly identifies noise/outliers (useful directly for anomaly
  detection — see the `Anomaly Detection with Clustering` folder).
- Robust to outliers (they just get labeled noise, don't distort centroids).

**Weaknesses**:
- Struggles when clusters have **very different densities** (a single global
  ε can't be locally dense in a sparse cluster and locally sparse in a dense
  one).
- Sensitive to the ε/minPts choice; performance degrades in high dimensions
  (curse of dimensionality makes distances less meaningful — the "empty
  space phenomenon").
- Not deterministic in edge cases: a border point that is reachable from
  two different clusters gets assigned to whichever cluster is processed
  first.

## 7. Basic → Pro
1. **Basic**: fixed ε/minPts on 2D data, visualize core/border/noise points.
2. **Intermediate**: use the k-distance plot to choose ε systematically.
3. **Pro**: **HDBSCAN** (Hierarchical DBSCAN) — removes the need to pick a
   single global ε by building a hierarchy over *varying* density
   thresholds and extracting the most stable clusters; handles
   variable-density clusters much better than vanilla DBSCAN.

## 8. Code in this folder
- `code/dbscan_scratch.py` — DBSCAN implemented from scratch in NumPy
  (region query, core/border/noise labeling, cluster expansion via BFS).
- `code/dbscan_sklearn_demo.py` — `sklearn.cluster.DBSCAN` with a k-distance
  plot for choosing ε, plus a comparison against K-Means on the two-moons
  dataset to show DBSCAN's arbitrary-shape advantage.
