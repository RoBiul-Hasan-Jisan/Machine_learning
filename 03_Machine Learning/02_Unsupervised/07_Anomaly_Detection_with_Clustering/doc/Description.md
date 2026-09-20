# Anomaly Detection with Clustering


##  Problem Definition

Anomaly detection has no single formal definition. Three distinct framings
are in common use, and each of the methods below commits to one of them
implicitly. Choosing a method without identifying which framing matches the
problem at hand is the most common source of poor results in practice.

| Framing | Formal statement | Method |
|---|---|---|
| Global density estimation | A point $x$ is anomalous if $p(x)$ is low under the estimated data distribution | GMM negative log-likelihood |
| Global distance | A point $x$ is anomalous if it is far from the nearest reference point (e.g., centroid) | K-Means centroid distance |
| Local density deviation | A point $x$ is anomalous if its local density is low relative to its neighborhood, independent of its absolute density | LOF |

These framings are not interchangeable. A point in a sparse but legitimate
region of a naturally low-density cluster is classified as anomalous under
the first two framings but not under the third. The choice of framing
should be made before the choice of algorithm.

##  Baseline Formulation

Clustering provides a model of "normal structure" over the data. Given such
a model, a point is a candidate anomaly if it is:

- distant from every cluster center,
- located in a low-density region, or
- explicitly labeled as noise by a density-based clustering procedure.

Any clustering algorithm can therefore be converted into an anomaly
detector by defining a scoring rule on top of its output. Sections 3–6
formalize four such scoring rules.

##  Method I: Centroid-Distance Scoring

###  Procedure

1. Fit K-Means with $k$ clusters on data assumed to be predominantly normal.
2. For each point $x$, compute $d(x, \mu_{c(x)})$, its distance to the
   centroid of its assigned cluster.
3. Flag points for which this distance exceeds a threshold, defined either
   as $\text{mean} + n \cdot \text{std}$ of the within-cluster distance
   distribution, or as the $p$-th percentile of that distribution.

###  Failure Mode: Clustered Anomalies

K-Means minimizes the objective

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} \lVert x - \mu_i \rVert^2$$

This objective contains no mechanism for distinguishing a legitimate
cluster from a cluster composed entirely of anomalies; it minimizes
within-cluster variance without reference to any external notion of
normality. Consequently, if a subset of anomalies is mutually close in
feature space, Lloyd's algorithm will assign them a dedicated centroid,
driving their within-cluster distance toward zero and rendering them
undetectable by this method. This is a direct consequence of the objective
function rather than a parameter-tuning failure, and is not resolved by
increasing $k$ or the number of random restarts.

###  Threshold Selection

The $\text{mean} + n \cdot \text{std}$ threshold assumes the within-cluster
distance distribution is approximately symmetric. Under skewed
distributions, this threshold either fails to flag genuine anomalies or
incorrectly flags normal points. The percentile-based threshold is
nonparametric and should be preferred unless the distance distribution has
been verified to be approximately Gaussian.

##  Method II: DBSCAN Noise Labeling

DBSCAN (see `DBSCAN/doc/README.md`) assigns the label `noise` (conventionally
`-1`) to points that do not meet its density-reachability criterion. No
additional scoring step is required to use this as an anomaly detector.

### Formal Basis

A point is classified as noise if and only if it is not density-reachable
from any core point, where a core point is defined as one with at least
`minPts` neighbors within radius $\varepsilon$. This is a connectivity
criterion on the $\varepsilon$-neighborhood graph rather than a density
estimate in the statistical sense.

###  Limitation

Because the criterion uses a single global value of $\varepsilon$, DBSCAN
cannot distinguish between a genuine anomaly and a normal point located in
a naturally sparse region of the data, provided both fail the same fixed
threshold. This limitation motivates Method III.

##  Method III: Local Outlier Factor

LOF addresses the global-threshold limitation of Method II by comparing a
point's local density to the local density of its neighbors, rather than to
a single fixed threshold.

###  Definitions

**k-distance.** For a point $x$, $\text{k-distance}(x)$ is the distance to
its $k$-th nearest neighbor.

**Reachability distance.**

$$\text{reach-dist}_k(x, o) = \max\big(\text{k-distance}(o),\ d(x, o)\big)$$

This construction bounds the reachability distance from below by
$\text{k-distance}(o)$, which stabilizes the score for points that are very
close to, or coincident with, $o$.

**Local reachability density.**

$$\text{lrd}_k(x) = \left( \frac{\sum_{o \in N_k(x)} \text{reach-dist}_k(x, o)}{|N_k(x)|} \right)^{-1}$$

**Local Outlier Factor.**

$$\text{LOF}_k(x) = \frac{\sum_{o \in N_k(x)} \text{lrd}_k(o)}{|N_k(x)| \cdot \text{lrd}_k(x)}$$

###  Interpretation

- $\text{LOF}(x) \approx 1$: the local density of $x$ is comparable to that
  of its neighbors; $x$ is classified as normal.
- $\text{LOF}(x) \gg 1$: the local density of $x$ is substantially lower
  than that of its neighbors; $x$ is classified as an outlier, regardless of
  whether it lies near an overall dense region of the data.

This relative formulation is what allows LOF to operate correctly on data
with regions of differing density, where a single global threshold (as in
Method II) is insufficient.

###  Limitation

Unlike DBSCAN's binary noise flag or the calibrated likelihood produced by
a GMM, LOF does not define a natural decision threshold. The choice of
cutoff above which a score is treated as anomalous is a tuning decision,
typically set by percentile or by manual review of flagged points.

##  Method IV: Gaussian Mixture Model Likelihood

A GMM models the data as a mixture of $k$ Gaussian components:

$$p(x) = \sum_{i=1}^{k} \pi_i \, \mathcal{N}(x \mid \mu_i, \Sigma_i)$$

fit by Expectation-Maximization. Points are flagged as anomalies when their
likelihood under the fitted mixture, $p(x)$, falls below a threshold; the
negative log-likelihood, $-\log p(x)$, serves as a continuous anomaly score.

###  Relationship to K-Means

Because each component's covariance $\Sigma_i$ may be a full matrix rather
than a scalar multiple of the identity, a GMM can represent elliptical
clusters of differing size and orientation — a class of structure that
K-Means's isotropic assumption cannot represent. This is the reason GMM
likelihood scoring tends to outperform centroid-distance scoring (Method I)
when the normal data does not form spherical, equally-sized clusters.

##  Evaluation Methodology

###  With Ground-Truth Labels

When even a small number of labeled anomalies is available, evaluation
should use precision, recall, ROC-AUC, or PR-AUC, treating the anomaly
class as positive. Because anomalies are typically the minority class,
raw accuracy is not informative. PR-AUC is generally preferred over
ROC-AUC under severe class imbalance, since ROC-AUC can remain high even
when precision on the minority class is poor.

### Without Ground-Truth Labels

In the absence of labels, flagged points should be reviewed manually, and
the detection threshold should be tuned to the volume of alerts that can
realistically be reviewed. This is a legitimate operational stopping
criterion rather than a methodological compromise.

##  Practical Considerations


- **Feature scaling.** All four methods rely on a distance or density
  computation; scores are not meaningful if features are on different
  scales. Standardization should precede any of these methods.
- **Class imbalance in evaluation splits.** Train/test splits should be
  stratified on the anomaly label when one exists, since an unstratified
  split can produce a test fold containing no anomalies.
- **Ensembling.** Because each method encodes a distinct formal definition
  of anomaly (Section 1), agreement between methods (e.g., a point flagged
  by both DBSCAN noise labeling and a high LOF score) constitutes stronger
  evidence than any single method's output, and combining signals is
  recommended for production systems.


##  Code Reference

| File | Contents |
|---|---|
| `code/anomaly_detection_scratch.py` | From-scratch implementations of centroid-distance scoring, DBSCAN noise-based scoring, and a simplified LOF |
| `code/anomaly_detection_sklearn_demo.py` | `sklearn.neighbors.LocalOutlierFactor` and `sklearn.cluster.DBSCAN` applied to synthetic data with injected outliers, with precision/recall evaluation against known ground truth |