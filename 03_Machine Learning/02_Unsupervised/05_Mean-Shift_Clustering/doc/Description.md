# Mean-Shift Clustering



##  Problem Framing

Mean-Shift treats the observed data as samples drawn from an unknown
probability density and defines clusters as the regions of attraction
around each local maximum (mode) of that density. Rather than fixing a
number of clusters or a connectivity threshold, the algorithm locates
these modes directly, for every point, by iteratively moving toward the
weighted average of nearby points — climbing the estimated density surface
by gradient ascent. The number of clusters is therefore not a parameter of
the method; it is a byproduct of however many distinct modes the estimated
density happens to contain.

##  Kernel Density Estimation

Mean-Shift implicitly estimates the data's underlying probability density
using kernel density estimation (KDE):

$$f(x) = \frac{1}{n} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right)$$

where $K$ is a kernel function — commonly Gaussian or flat/uniform — and
$h$ is the bandwidth, which controls the degree of smoothing applied to
the density estimate. The bandwidth is the single most consequential
hyperparameter in the method (Section 6), since it determines the scale
at which density variation is resolved.

##  The Mean-Shift Vector

Taking the gradient of the KDE estimate in Section 2 with respect to $x$,
and simplifying for a Gaussian (or, more generally, a radially symmetric)
kernel, yields the **mean-shift vector**:

$$m(x) = \frac{\sum_i x_i \, K\!\left(\frac{x - x_i}{h}\right)}{\sum_i K\!\left(\frac{x - x_i}{h}\right)} - x$$

This expression is the weighted mean of nearby points minus the point's
current position, with weights given by the kernel. Because $m(x)$ is
derived directly from the gradient of $f$, it points in the direction of
steepest increase in the estimated density at $x$; consequently, moving a
point along $m(x)$ is equivalent to performing gradient ascent on the
density surface, and repeating this step drives the point toward the
nearest local mode.

##  Algorithm Specification

**Input:** dataset $X$, kernel $K$, bandwidth $h$.

1. Initialize a candidate point at the position of each data point.
2. For each candidate, repeat until the shift magnitude falls below a
   convergence threshold:
   - Compute the weighted mean of all points within bandwidth $h$ of the
     candidate, using kernel-derived weights (Section 3).
   - Move the candidate to this weighted mean.
3. Group candidates that converge to (approximately) the same location.
   Each distinct convergence location is a cluster mode.
4. Assign each original data point to the cluster corresponding to the
   mode its candidate converged to.

Because every candidate converges to some mode under this procedure
(barring numerical non-convergence), every point receives a cluster
assignment; unlike DBSCAN, Mean-Shift does not produce an explicit noise
label .

##  Complexity Analysis

The naive algorithm requires $O(n^2)$ time per iteration, since each
candidate must evaluate its distance to every other point to determine
which fall within bandwidth $h$, and this cost is incurred at every
iteration until convergence. This makes the naive implementation expensive
for large $n$.

**Practical acceleration.** Two techniques are used to reduce this cost:

- Spatial indexing (KD-tree or Ball-tree, see
  `Efficient Tree-based Algorithms/doc/Description.md`) to accelerate the
  within-bandwidth neighbor query at each iteration.
- Bin seeding (`bin_seeding=True` in `scikit-learn`), which discretizes
  the data onto a coarse grid and runs the full shifting procedure only
  from a representative subset of candidate starting points, rather than
  from every data point.

##  Bandwidth Selection

The bandwidth $h$ is the dominant factor determining the outcome of the
algorithm, in direct analogy to DBSCAN's single global $\varepsilon$
parameter (see `DBSCAN/doc/Description.md`, Section 6):

- **$h$ too small.** The density estimate becomes noisy and spiky, with
  many spurious local maxima, producing a large number of small, spurious
  clusters.
- **$h$ too large.** The density estimate is over-smoothed, and distinct
  modes merge into a single broad peak, producing too few clusters — in
  the extreme, a single cluster covering the entire dataset.

**Practical heuristic.** `sklearn.cluster.estimate_bandwidth()` estimates
a reasonable bandwidth from the data by computing a quantile of pairwise
nearest-neighbor distances, providing a data-driven starting point rather
than requiring the value to be chosen arbitrarily.

##  Strengths and Limitations

**Strengths.**
- Does not require the number of clusters to be specified; it is
  determined by the number of density modes discovered (Section 1).
- Recovers clusters of arbitrary shape, within the scale set by the
  bandwidth, since the procedure follows the local density gradient rather
  than assuming any particular cluster geometry.
- Robust to outliers in the sense that low-density points do not
  contribute to forming, or converge toward, a strong mode.

**Limitations.**
- Computationally expensive (Section 5); does not scale well to large $n$
  or to high-dimensional data, where density estimation itself becomes
  less reliable due to the curse of dimensionality.
- Highly sensitive to the bandwidth parameter $h$ (Section 6).
- Struggles with clusters of substantially different density or scale,
  since a single global bandwidth cannot simultaneously resolve a dense
  cluster and a sparse one without either fragmenting the sparse cluster
  or over-merging the dense one — the same structural issue that limits
  DBSCAN's single global $\varepsilon$ (see
  `DBSCAN/doc/Description.md`, Section 6).

##  Relation to Other Methods

**Comparison with DBSCAN.** Both methods are density-based and do not
require the number of clusters to be specified in advance. They differ in
mechanism and in output: DBSCAN determines cluster membership through a
connectivity criterion over an $\varepsilon$-neighborhood graph and
explicitly labels low-density points as noise (see
`DBSCAN/doc/Description.md`, Section 2), whereas Mean-Shift determines
cluster membership through gradient ascent toward a density mode and
assigns every point to some cluster, with no explicit noise label.

**Relation to CAMShift.** Mean-Shift is the clustering mechanism
underlying the CAMShift (Continuously Adaptive Mean-Shift) algorithm, a
widely used method for object tracking in computer vision, where the mode
being tracked corresponds to the location of a target object's color
distribution in successive video frames.


##  Code Reference

| File | Contents |
|---|---|
| `code/meanshift_scratch.py` | Mean-Shift implemented from scratch in NumPy with a Gaussian kernel, convergence tracking, and mode merging |
| `code/meanshift_sklearn_demo.py` | `sklearn.cluster.MeanShift` with automatic bandwidth estimation, visualized on blob data |