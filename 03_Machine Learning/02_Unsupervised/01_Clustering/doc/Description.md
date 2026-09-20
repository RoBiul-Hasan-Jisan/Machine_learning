# Clustering


##  Problem Definition

Clustering is an unsupervised learning task: given data
$X = \{x_1, \dots, x_n\}$ with no associated labels, find an assignment
$c: X \rightarrow \{1, \dots, k\}$ such that points assigned to the same
group are more similar to one another than to points assigned to
different groups. Formally, this requires optimizing some notion of
intra-cluster similarity and inter-cluster dissimilarity, but no labeled
ground truth exists against which to check the result directly. Success
is therefore judged either by internal structural criteria — compactness
and separation, evaluated on the data alone (Section 6) — or, when labels
exist solely for the purpose of evaluation and were withheld from the
algorithm, by external agreement metrics (Section 6).

##  The Impossibility of a Universal Clustering Objective

A foundational theoretical result, due to Kleinberg (2002), establishes
that no clustering function can simultaneously satisfy three properties
that each appear independently desirable:

- **Scale-invariance.** The clustering result should not change if all
  pairwise distances are scaled by a positive constant.
- **Richness.** For any partition of the data into clusters, there should
  exist some distance function under which the clustering function
  returns that exact partition — that is, the function must be capable of
  producing any possible clustering, given suitable input.
- **Consistency.** Shrinking intra-cluster distances and/or growing
  inter-cluster distances, while leaving the existing cluster assignment
  otherwise unchanged, should not cause the clustering function to change
  its output.

Kleinberg proves that no function satisfies all three simultaneously.

**Consequence.** Every clustering algorithm surveyed in Section 5
implicitly sacrifices one of these properties, and different algorithms
sacrifice different ones. K-Means's fixed-$k$ requirement and
variance-minimizing objective, DBSCAN's density-threshold definition of a
cluster, and hierarchical clustering's linkage-dependent merge criterion
each encode a distinct, mutually incompatible notion of what a cluster
is. This is the formal justification, rather than merely a practical
observation, for the repeated recommendation throughout this repository
to try multiple algorithm families on a given dataset (Section 8): the
disagreement between algorithms on ambiguous data is not a symptom of
implementation error, but a direct consequence of Kleinberg's result.

##  Applications

Clustering is used across a broad range of tasks that share the common
requirement of discovering structure without labeled examples:

- Customer segmentation, in marketing analytics.
- Anomaly and fraud detection, where a point's distance from every
  cluster or its assignment to a low-density region serves as an outlier
  score (see `Anomaly Detection with Clustering/doc/Description.md`).
- Document and topic grouping.
- Image compression via color quantization.
- Gene expression grouping in computational biology.
- A preprocessing step ahead of supervised learning, when labels are
  unavailable or when structure discovered by clustering is itself a
  useful feature.

##  Distance and Similarity Metrics

The choice of distance metric encodes an assumption about the geometry
of the feature space, and this assumption is inherited by every
algorithm built on top of that metric.

| Metric | Definition | Assumption encoded |
|---|---|---|
| Euclidean | $d(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$ | Isotropic geometry — every dimension contributes equally and independently to distance |
| Manhattan | $d(x, y) = \sum_i \lvert x_i - y_i \rvert$ | Less sensitive to large deviations in a single dimension than Euclidean distance, owing to the absence of squaring |
| Cosine similarity | $\dfrac{x \cdot y}{\lVert x \rVert \lVert y \rVert}$ | Magnitude is irrelevant; only direction matters — appropriate for text or other high-dimensional sparse data where vector length reflects incidental factors (e.g., document length) rather than semantic content |
| Mahalanobis | $d(x, y) = \sqrt{(x-y)^\top \Sigma^{-1} (x-y)}$ | Accounts for feature correlation via the covariance matrix $\Sigma$; reduces to Euclidean distance when $\Sigma = I$ |

**High-dimensional distance concentration.** As dimensionality $d$
increases, the ratio $\frac{d_{\max} - d_{\min}}{d_{\min}}$ between the
farthest and nearest pairwise distances tends toward zero for many data
distributions (see
`Dimensionality Reduction — Overview/doc/Description.md`, Section 2).
This degrades every distance-based method surveyed in Section 5, since
distance becomes a progressively weaker signal of true similarity as
dimensionality grows — a consideration that motivates dimensionality
reduction as a preprocessing step for clustering on high-dimensional data.

**K-Means and the isotropic-Gaussian assumption.** K-Means's use of
squared Euclidean distance to a single centroid is mathematically
equivalent to assuming each cluster is generated by an isotropic Gaussian
distribution with equal variance across clusters (see
`K-Means/doc/Description.md`, Section 7). When true cluster shapes are
elongated or have unequal variance, K-Means's objective is a provably
poor fit to the data, regardless of how $k$ is tuned — this is a
consequence of the objective function's form, not a limitation that
better parameter selection can resolve.

##  Algorithm Families

| Family | Defining idea | Members |
|---|---|---|
| Centroid-based | Minimize distance to a representative center point | K-Means, Mean-Shift |
| Density-based | A cluster is a dense region separated from other dense regions by sparse regions | DBSCAN |
| Hierarchical | Construct a complete nested sequence of partitions, represented as a tree | Agglomerative, Divisive |
| Graph-based | Represent data as a similarity graph and cluster via its spectral (eigenvalue/eigenvector) properties | Spectral Clustering |
| Distribution-based | Assume data is generated by a mixture of parametric probability distributions | Gaussian Mixture Models |

Each family embodies a distinct choice among the properties addressed by
Kleinberg's theorem (Section 2), and each has a dedicated theory document
in this repository developing its formal objective, algorithm
specification, and complexity analysis in full: see
`K-Means/doc/Description.md`, `DBSCAN/doc/Description.md`,
`Hierarchical Clustering/doc/Description.md`,
`Mean-Shift/doc/Description.md`, and
`Spectral Clustering/doc/Description.md`.

##  Evaluation Methodology

**Internal metrics (no labels required).**

- **Inertia / within-cluster sum of squares (WCSS).** The quantity
  K-Means minimizes directly (see `K-Means/doc/Description.md`, Section
  2); lower values indicate tighter clusters. Because inertia decreases
  monotonically as the number of clusters increases — reaching zero when
  every point is its own cluster — it cannot be minimized directly to
  choose $k$; it is instead used with the elbow heuristic (Section 7).
- **Silhouette score.** For each point, defined as
  $\frac{b - a}{\max(a, b)}$, where $a$ is the mean intra-cluster distance
  and $b$ is the mean distance to the nearest other cluster. The score
  lies in $[-1, 1]$, with values near 1 indicating that a point is well
  matched to its own cluster and poorly matched to neighboring clusters.
  Because both $a$ and $b$ are computed from raw distances, this score is
  biased toward rewarding convex cluster shapes: a correctly identified
  non-convex cluster, such as one recovered by DBSCAN, can receive a
  lower silhouette score than an incorrect convex split of the same data.
  Silhouette score is therefore best used to tune hyperparameters within a
  single algorithm family rather than to compare across algorithm
  families with different underlying shape assumptions.
- **Davies–Bouldin index.** The ratio of within-cluster scatter to
  between-cluster separation, averaged over clusters; lower values
  indicate better-separated, more compact clustering.

**External metrics (labels available only for evaluation).**

- **Adjusted Rand Index (ARI).** Measures agreement between the
  clustering and ground-truth labels, corrected for the level of
  agreement expected by chance; symmetric and the standard choice for
  comparing a clustering result to ground truth.
- **Normalized Mutual Information (NMI).** Measures shared information
  between the clustering and the ground-truth labeling. Uncorrected NMI
  can be inflated by clusterings containing many small clusters; the
  adjusted variant (AMI) corrects for this and should generally be
  preferred over plain NMI when comparing clusterings with different
  numbers of clusters.
- **Homogeneity and completeness.** Homogeneity measures whether each
  cluster contains only members of a single ground-truth class;
  completeness measures whether all members of a given ground-truth class
  are assigned to the same cluster. The two are complementary and are
  typically reported together, since optimizing either alone admits a
  trivial solution (a single cluster achieves perfect completeness; every
  point as its own cluster achieves perfect homogeneity).

##  Implementation Roadmap


**Choosing k or an equivalent parameter.** The elbow method (plotting
inertia against $k$ and identifying the point of diminishing returns) and
silhouette analysis (Section 6) are the standard heuristics used
throughout this repository's method documents, and are addressed in
detail in `K-Means/doc/Description.md`, Section 4. The gap statistic
(Tibshirani, Walther, and Hastie, 2001) is a more principled alternative,
comparing observed inertia against the inertia expected under an
appropriate null reference distribution, and is a natural extension for
readers who find the elbow heuristic's ambiguity limiting.

##  Practical Considerations

- **Feature scaling.** Distance-based clustering methods require features
  to be standardized beforehand; otherwise, a feature with a larger
  numeric range dominates the distance computation regardless of its
  actual relevance (see Section 4's discussion of Euclidean distance's
  isotropy assumption). This applies to every method in Section 5 that
  relies on a distance metric.
- **No single best algorithm.** As established formally in Section 2,
  Kleinberg's impossibility theorem guarantees that different clustering
  algorithms will disagree on data whose structure does not clearly favor
  one algorithm's implicit assumptions. Evaluating multiple algorithm
  families on the same data, rather than committing to a single method,
  is therefore a principled response to this result rather than a
  hedge against uncertainty.
- **Visualization for high-dimensional data.** When data has more than
  two or three dimensions, visualizing clustering results directly is not
  possible; PCA or t-SNE (see
  `Dimensionality Reduction — Overview/doc/Description.md`) are the
  standard tools for producing a two-dimensional projection for visual
  inspection, with the caveat that t-SNE's local-structure-preserving
  objective means inter-cluster distances and apparent cluster sizes in
  such a projection are not reliable (see `t-SNE/doc/Description.md`,
  Section 7).
- **Ensemble and consensus clustering.** Because Kleinberg's theorem
  implies no single algorithm can be considered correct in general,
  combining multiple clusterings of the same data — via consensus or
  ensemble methods — is a further principled response for obtaining a
  more stable result than any single algorithm provides in isolation.

