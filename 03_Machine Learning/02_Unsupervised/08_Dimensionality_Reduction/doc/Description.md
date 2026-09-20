# Dimensionality Reduction


## 1. Problem Definition

Given data $X \in \mathbb{R}^{n \times d}$, dimensionality reduction seeks
a mapping $f: \mathbb{R}^d \rightarrow \mathbb{R}^k$ with $k \ll d$, such
that the low-dimensional representation $f(X)$ retains as much of the
structure of $X$ that is relevant to a downstream task as possible. What
"relevant structure" means is method-dependent — variance, pairwise
geodesic distance, local neighborhood relationships, or neighborhood
probabilities — and this choice is the primary axis along which the
methods below differ.

Two distinct objectives are commonly pursued under this same formal
definition, and they are frequently in tension:

- **Compression and denoising.** Retain most of the variance or
  information content of the data using fewer dimensions, typically as a
  preprocessing step before storage, modeling, or noise removal.
- **Visualization.** Project to two or three dimensions specifically to
  inspect cluster or manifold structure visually, even at the cost of
  distorting exact distances.

##  The Curse of Dimensionality

High-dimensional data presents three related theoretical obstacles that
motivate dimensionality reduction as a preprocessing step:

1. **Distance concentration.** As dimensionality $d$ increases, the ratio
   $\frac{d_{\max} - d_{\min}}{d_{\min}}$ between the farthest and nearest
   neighbor distances tends toward zero for many data distributions. This
   means pairwise distances become less discriminative, which directly
   degrades any downstream method — including clustering and
   nearest-neighbor search — that relies on distance as a notion of
   similarity.
2. **Sample complexity.** The volume of the space grows exponentially with
   $d$, so the number of observations required to maintain a fixed sampling
   density grows exponentially as well. Models trained on high-dimensional
   data with fixed sample size therefore generalize worse, all else equal.
3. **Redundancy and noise accumulation.** Additional features are not
   guaranteed to carry additional signal; in practice many features are
   correlated or uninformative, and their inclusion adds variance to
   downstream estimates without a compensating gain in information.

##  Linear versus Manifold Methods

| Class | Structural assumption | Methods |
|---|---|---|
| Linear | The data lies in, or near, a linear subspace of $\mathbb{R}^d$ | PCA, NMF |
| Manifold (non-linear) | The data lies on a curved lower-dimensional manifold embedded in $\mathbb{R}^d$ | Isomap, LLE, t-SNE, Spectral Embedding |

**Canonical illustration.** The Swiss-roll dataset — a two-dimensional
sheet rolled into a three-dimensional spiral — is linearly inseparable
from its ambient space in the relevant sense: any linear projection, such
as PCA, necessarily conflates points that are close in ambient (straight-
line) distance but far apart along the manifold's intrinsic geometry.
Manifold methods such as Isomap and LLE are constructed specifically to
recover the intrinsic two-dimensional structure by using geodesic or
locally-linear relationships instead of straight-line distance.

##  Method Survey

The following methods, ordered by increasing structural flexibility, are
covered in dedicated folders within this repository.

1. **Principal Component Analysis (PCA)** — linear. Projects data onto the
   orthogonal directions of maximum variance. Serves as the baseline
   against which non-linear methods are typically compared, and is the
   only method in this survey with a simple closed-form solution (via
   eigendecomposition of the covariance matrix, or equivalently the SVD of
   the centered data matrix).
2. **Non-negative Matrix Factorization (NMF)** — linear, with an
   additional non-negativity constraint on both factors. This constraint
   forecloses subtractive combinations of components, which tends to yield
   parts-based, additive representations — a property exploited in
   applications such as topic modeling and image decomposition.
3. **Isomap** — non-linear, global. Preserves geodesic (along-manifold)
   distance rather than Euclidean distance, by constructing a neighborhood
   graph and using shortest-path distances on that graph as an
   approximation to geodesic distance, followed by classical multidimensional
   scaling.
4. **Locally Linear Embedding (LLE)** — non-linear, local. Reconstructs
   each point as a linear combination of its nearest neighbors in the
   original space, then finds a low-dimensional embedding that preserves
   those same reconstruction weights. This makes LLE sensitive to local
   neighborhood structure but, unlike Isomap, does not explicitly preserve
   global distances.
5. **t-SNE** — non-linear, visualization-oriented. Converts pairwise
   distances into conditional probabilities of neighbor selection in both
   the high- and low-dimensional spaces, then minimizes the
   Kullback–Leibler divergence between the two probability distributions.
   This objective is optimized for preserving local neighborhood structure
   at the expense of global distance fidelity, which is why it is
   well-suited to visualization but not to tasks requiring quantitatively
   meaningful distances in the embedding.
6. **Spectral Embedding** — the dimensionality-reduction step that
   underlies Spectral Clustering (see `../Spectral Clustering`). It
   computes the eigenvectors of a graph Laplacian derived from a
   similarity graph over the data; the resulting embedding is then
   typically clustered with K-Means. Spectral Clustering is thus best
   understood as spectral embedding used specifically for the clustering
   task, rather than as a distinct family of technique.

##  Method Selection Criteria

The appropriate method follows directly from which structural assumption
(Section 3) and which downstream requirement applies:

| Requirement | Method |
|---|---|
| An interpretable, invertible linear transform, or output feeding into another linear model | PCA |
| Strictly non-negative, parts-based factors (e.g., topics, facial components) | NMF |
| Preservation of true manifold (geodesic) distance for a physically meaningful embedding | Isomap |
| Preservation of local neighborhood structure with minimal small-scale distortion | LLE |
| A two- or three-dimensional visualization of cluster structure, without requiring exact distance preservation | t-SNE (or UMAP, which shares this objective and is typically faster, though not covered in this repository) |

##  Evaluation Methodology

The appropriate evaluation metric depends on whether the method admits an
explicit inverse mapping back to the original space.

- **Explained variance ratio** (PCA-specific): the fraction of the
  original data's total variance captured by the retained components.
  Well-defined only because PCA's components are ranked by the variance
  they capture, a property not shared by the other methods in this survey.
- **Reconstruction error**: the discrepancy between the original data and
  its reconstruction from the low-dimensional representation. Meaningful
  only for methods with an explicit inverse transform, such as PCA and
  NMF; not applicable to Isomap, LLE, or t-SNE, none of which define an
  inverse mapping.
- **Trustworthiness / neighborhood preservation**: quantifies the degree
  to which the $k$-nearest neighbors of a point in the low-dimensional
  embedding match its $k$-nearest neighbors in the original space
  (`sklearn.manifold.trustworthiness`). This is the standard evaluation
  criterion for manifold-learning and visualization methods such as t-SNE,
  where no reconstruction error is available, since it directly measures
  the property those methods are designed to preserve.

##  Code Reference

| File | Contents |
|---|---|
| `code/dimensionality_reduction_overview.py` | Runnable side-by-side comparison of PCA, Isomap, LLE, and t-SNE embedding the Swiss-roll and digits datasets into two dimensions, illustrating how each method distorts the data differently |

Dedicated from-scratch implementations and method-specific theoretical
treatments are provided in their own folders: `Principal Component
Analysis/`, `Non-negative Matrix Factorization/`, `Isomap/`, `Locally
Linear Embedding/`, and `t-SNE/`.