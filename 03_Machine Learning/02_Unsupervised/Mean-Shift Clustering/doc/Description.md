# Mean-Shift Clustering

## 1. Intuition
Mean-Shift is a **mode-seeking** algorithm: think of your data points as
samples from an unknown probability density. Mean-Shift finds the local
**peaks (modes)** of that density by, for every point, repeatedly shifting it
towards the average of its neighbors — literally "shifting toward the mean."
Points that converge to the same peak belong to the same cluster.

No need to specify the number of clusters — it falls out naturally from how
many distinct density peaks exist.

## 2. Kernel Density Estimation (KDE) — the foundation
Mean-Shift implicitly estimates the data's probability density using KDE:

```
f(x) = (1/n) * Σ_i K((x - x_i) / h)
```

where `K` is a kernel (commonly the **Gaussian** or **flat/uniform** kernel)
and `h` is the **bandwidth** (controls smoothing — this is the single most
important hyperparameter).

## 3. The mean-shift vector
Taking the gradient of the KDE estimate and simplifying (for a Gaussian
kernel) gives the **mean shift vector**, which always points in the
direction of the steepest increase in density:

```
m(x) = ( Σ_i x_i * K((x - x_i)/h)  /  Σ_i K((x - x_i)/h) ) - x
```

This is exactly "the weighted mean of nearby points, minus my current
position" — moving along `m(x)` climbs the density surface uphill.

## 4. Algorithm
1. For every data point, initialize a "shifting" candidate point at that
   position.
2. Repeat until convergence (shift becomes negligible):
   - Compute the weighted mean of all points within bandwidth `h` of the
     candidate (weights from the kernel, e.g. Gaussian falloff).
   - Move the candidate to that weighted mean.
3. Points that converge to (nearly) the same location are grouped: each
   distinct convergence point is a cluster mode/centroid.
4. Assign each original point to the cluster of the mode it converged to.

## 5. Complexity
- Naive: `O(n^2)` per iteration (each point must look at all others within
  bandwidth), times number of iterations — expensive for large `n`.
- Sped up in practice with spatial indexing (KD-tree/Ball-tree) and by
  "binning" points into a coarse grid (`sklearn`'s `bin_seeding=True`) so
  only a subset of representative points need to run the full shifting
  procedure.

## 6. Choosing the bandwidth `h`
This is the single biggest lever:
- Too small → many tiny clusters (density estimate too noisy/spiky).
- Too large → everything merges into one cluster (over-smoothed density).
- Practical heuristic: `sklearn.cluster.estimate_bandwidth()` estimates a
  reasonable bandwidth from pairwise distances (quantile of nearest-neighbor
  distances).

## 7. Strengths / Weaknesses
**Strengths**:
- No need to choose the number of clusters.
- Finds clusters of arbitrary shape (limited by bandwidth-scale locality).
- Robust to outliers (low-density points don't form/join a strong mode).

**Weaknesses**:
- Computationally expensive, doesn't scale well to large or high-dimensional
  data.
- Very sensitive to bandwidth choice.
- Struggles with clusters of very different density/scale (a single global
  bandwidth, similar issue to DBSCAN's single global ε).

## 8. Relation to other methods
- Like DBSCAN, it's density-based and doesn't need `k` — but Mean-Shift
  gives every point a cluster (no explicit "noise" label) and works by
  gradient ascent rather than connectivity/reachability.
- It's the clustering backbone of the classic **CAMShift** object-tracking
  algorithm in computer vision.

## 9. Basic → Pro
1. **Basic**: flat/uniform kernel, fixed bandwidth, visualize convergence
   paths.
2. **Intermediate**: Gaussian kernel + `estimate_bandwidth` for automatic
   bandwidth selection.
3. **Pro**: bin-seeding / grid-based candidate reduction for scalability;
   compare against **Quick Shift** (a related mode-seeking algorithm used
   in image segmentation).

## 10. Code in this folder
- `code/meanshift_scratch.py` — Mean-Shift implemented from scratch in NumPy
  with a Gaussian kernel, convergence tracking, and mode merging.
- `code/meanshift_sklearn_demo.py` — `sklearn.cluster.MeanShift` with
  automatic bandwidth estimation, visualized on blob data.
