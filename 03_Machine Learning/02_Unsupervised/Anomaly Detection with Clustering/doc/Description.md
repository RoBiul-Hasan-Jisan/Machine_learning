# Anomaly Detection with Clustering

## 1. Intuition
Anomalies (outliers) are points that don't fit the normal structure of the
data. Clustering gives us a natural "normal structure" — anomalies are then
points that are: far from every cluster centroid, in a very sparse/low-density
region, or explicitly labeled noise by a density-based method. This folder
shows several clustering-based strategies for turning any clustering
algorithm into an outlier detector.

## 2. Strategy 1 — Distance-to-centroid (K-Means-based)
1. Run K-Means with `k` clusters on (mostly normal) data.
2. For every point, compute its distance to its assigned cluster's centroid.
3. Points whose distance exceeds a threshold (e.g. mean + `n` standard
   deviations, or the top `p`th percentile of distances) are flagged as
   anomalies.

Works well when normal data is genuinely blob-shaped, but is sensitive to
how many clusters you pick and can be fooled by a small tight cluster of
anomalies (which would just become its own "normal" cluster).

## 3. Strategy 2 — DBSCAN noise points
DBSCAN (see `DBSCAN/doc/README.md`) already labels sparse points as
**noise** (`label == -1`) as part of its normal operation — no extra step
needed! This is often a more natural anomaly detector than K-Means-distance
because it doesn't assume anomalies form their own cluster; it only requires
that they don't have `minPts` neighbors nearby.

## 4. Strategy 3 — Local Outlier Factor (LOF)
LOF improves on a single global density/distance threshold by comparing each
point's **local density** to the local density of its neighbors:

```
LOF(x) ≈ (average local density of x's k-nearest-neighbors) / (local density of x)
```

- `LOF(x) ≈ 1` → similar density to neighbors → normal.
- `LOF(x) >> 1` → much sparser than its neighbors → outlier (even if it is
  near a dense cluster region overall — this handles varying-density data
  much better than a single global threshold).

## 5. Strategy 4 — Gaussian Mixture Models (probabilistic)
Fit a Gaussian Mixture Model (soft clustering — see `K-Means/doc/README.md`
"soft clustering" note) and flag points with low likelihood under the fitted
mixture density as anomalies. This gives a continuous "anomaly score"
(negative log-likelihood) rather than a hard in/out decision.

## 6. Evaluating anomaly detectors
- If you have some labeled anomalies (even a handful), use precision/recall
  or ROC-AUC / PR-AUC treating "anomaly" as the positive class (anomalies
  are almost always the rare/minority class, so accuracy is misleading —
  precision/recall or PR-AUC matter more than accuracy).
- Without labels: sanity-check flagged points manually, tune the threshold
  by how many anomalies you can realistically review.

## 7. Practical tips
- **Always scale features first** — distance-based anomaly scores are
  meaningless if features have wildly different units/scales.
- Anomaly detection is inherently about **imbalance**: anomalies are rare.
  Don't split train/test naively without stratifying on the (rare) known
  anomaly label if you have one.
- Combine multiple signals (e.g. DBSCAN noise flag AND high LOF score) for
  more robust detection in production.

## 8. Basic → Pro
1. **Basic**: K-Means + distance-to-centroid threshold.
2. **Intermediate**: DBSCAN noise points, k-distance-based threshold tuning.
3. **Pro**: LOF for varying-density data; ensembling multiple detectors;
   compare against non-clustering approaches like **Isolation Forest** or
   **One-Class SVM** (not clustering-based, but the natural next step for a
   production anomaly-detection system).

## 9. Code in this folder
- `code/anomaly_detection_scratch.py` — from-scratch distance-to-centroid
  (K-Means-based) and DBSCAN-noise-based anomaly scoring, plus a from-scratch
  simplified LOF implementation.
- `code/anomaly_detection_sklearn_demo.py` — `sklearn.neighbors.LocalOutlierFactor`
  and `sklearn.cluster.DBSCAN`-based detection on synthetic data with injected
  outliers, with precision/recall evaluation against the known injected
  anomalies.
