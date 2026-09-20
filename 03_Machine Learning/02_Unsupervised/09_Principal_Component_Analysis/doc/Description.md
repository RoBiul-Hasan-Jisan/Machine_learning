# Principal Component Analysis (PCA)

##  Problem Definition

Given centered data $X \in \mathbb{R}^{n \times d}$ (each column with mean
zero), PCA seeks a sequence of orthogonal directions
$v_1, v_2, \dots, v_d \in \mathbb{R}^d$, called principal components, such
that $v_1$ is the direction along which the projected data has maximum
variance, $v_2$ is the direction of next-largest variance subject to being
orthogonal to $v_1$, and so on. Retaining only the first $k \ll d$
components yields a lower-dimensional representation that preserves as
much of the variance in $X$ as any $k$-dimensional linear projection can.

##  Derivation of the Principal Components

**First component.** The first principal component is defined as the unit
vector $v_1$ maximizing the variance of the projected data $Xv_1$:

$$v_1 = \arg\max_{\lVert v \rVert = 1} \operatorname{Var}(Xv) = \arg\max_{\lVert v \rVert = 1} v^\top C v$$

where $C = \frac{1}{n-1} X^\top X$ is the $d \times d$ sample covariance
matrix (well-defined without a further mean-subtraction term, since $X$ is
already centered by assumption).

**Solving via Lagrange multipliers.** Introducing a multiplier $\lambda$
for the constraint $v^\top v = 1$, the Lagrangian is
$\mathcal{L}(v, \lambda) = v^\top C v - \lambda (v^\top v - 1)$. Setting
the gradient with respect to $v$ to zero gives:

$$2Cv - 2\lambda v = 0 \quad \Longrightarrow \quad Cv = \lambda v$$

This shows that any stationary point of the constrained maximization must
be an eigenvector of $C$. Substituting $Cv = \lambda v$ back into the
objective gives $v^\top C v = \lambda v^\top v = \lambda$, so the
objective value at a stationary point equals its corresponding
eigenvalue. Since $C$ is symmetric positive semi-definite, all its
eigenvalues are real and non-negative, and the maximum of $v^\top C v$
over unit vectors is achieved by the eigenvector associated with the
largest eigenvalue $\lambda_1$.

**Subsequent components.** Each subsequent component $v_i$ is defined as
the unit vector maximizing $v^\top C v$ subject to orthogonality with all
previously chosen components, $v_1, \dots, v_{i-1}$. Because $C$ is
symmetric, its eigenvectors can be chosen to be mutually orthogonal (the
spectral theorem), and it follows by the same Lagrangian argument, applied
within the subspace orthogonal to the previously chosen components, that
$v_i$ is the eigenvector of $C$ associated with the $i$-th largest
eigenvalue $\lambda_i$. This establishes, rather than merely asserts, that
the principal components are precisely the eigenvectors of $C$ ordered by
decreasing eigenvalue, and that they are mutually orthogonal by
construction rather than by an additional imposed constraint.

**Projection and reconstruction.** With $V_k$ denoting the $d \times k$
matrix whose columns are the top-$k$ eigenvectors, the reduced
representation is $Z = X V_k$, and the corresponding approximate
reconstruction is $X_{\text{approx}} = Z V_k^\top$.

##  Equivalence with Singular Value Decomposition

PCA is computed in practice via the singular value decomposition
$X = U \Sigma V^\top$, rather than by eigendecomposing $C$ directly.

**Why these coincide.** Substituting the SVD of $X$ into the definition of
the covariance matrix:

$$C = \frac{1}{n-1} X^\top X = \frac{1}{n-1} V \Sigma^\top U^\top U \Sigma V^\top = \frac{1}{n-1} V \Sigma^2 V^\top$$

using $U^\top U = I$. This is exactly the eigendecomposition of $C$, with
eigenvectors given by the columns of $V$ and eigenvalues given by
$\lambda_i = \sigma_i^2 / (n-1)$, where $\sigma_i$ is the $i$-th singular
value of $X$. The columns of $V$ are therefore identical to the principal
components derived in Section 2, and no separate computation of $C$ is
required to obtain them.

**Why SVD is preferred numerically.** Forming $C = X^\top X$ explicitly
squares the condition number of the underlying data matrix, since the
singular values of $C$ are the squares of the singular values of $X$.
Computing the SVD of $X$ directly avoids this loss of numerical precision,
which is why `scikit-learn`'s implementation of PCA operates on $X$ via
SVD rather than on the explicitly formed covariance matrix.

## Explained Variance

The fraction of total variance captured by the $i$-th component is:

$$\text{explained\_variance\_ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

This quantity follows directly from Section 2: because $\lambda_i$ is
both the eigenvalue of $C$ associated with $v_i$ and the variance of the
data projected onto $v_i$, the ratio above is precisely the share of total
variance attributable to that direction. The cumulative explained variance
as a function of the number of retained components (the scree plot) is
the standard basis for choosing $k$ — for instance, retaining enough
components to reach a threshold such as 95% of total variance.

##  Preprocessing: Centering and Scaling

**Centering is required, not optional.** The derivation in Section 2
defines variance as $v^\top C v$ with $C$ computed from centered data;
if $X$ is not centered, $X^\top X$ conflates variance around the mean with
the squared magnitude of the mean itself, and the resulting "principal
components" no longer correspond to directions of variance in the sense
Section 2 establishes.

**Scaling is a modeling choice, not a mathematical requirement, but is
usually necessary in practice.** Because $C$ is computed directly from the
raw (centered) feature values, a feature with a numerically larger scale
— for instance, one measured in kilometers rather than meters — contributes
disproportionately to $C$ and therefore disproportionately to the
top principal components, independent of whether that feature carries
more genuine information. Standardizing each feature to unit variance
before applying PCA is the standard remedy when features are measured in
different units or scales.

##  Complexity Analysis

| Approach | Complexity |
|---|---|
| Covariance matrix construction + eigendecomposition | $O(n d^2 + d^3)$ |
| Direct SVD of $X$ | $O(\min(n d^2,\, n^2 d))$ |

SVD-based PCA scales more favorably than the covariance route when either
$d \gg n$ or $n \gg d$, and for large datasets, truncated or randomized
SVD algorithms can compute only the top-$k$ singular vectors directly,
avoiding the cost of computing the full decomposition when $k \ll d$.

##  Strengths and Limitations

**Strengths.**
- Deterministic: unlike K-Means or t-SNE, PCA involves no random
  initialization, so its result depends only on the input data.
- Possesses an exact linear inverse transform (Section 2), which makes it
  directly usable for compression and denoising, unlike methods such as
  Isomap, LLE, or t-SNE, none of which define an inverse mapping (see
  `Dimensionality Reduction — Overview/doc/Description.md`, Section 6).
- Components are mathematically well-defined as variance-maximizing
  directions (Section 2) and are orthogonal to one another by
  construction, not by an additional imposed constraint.

**Limitations.**
- Strictly linear: PCA can only represent variance captured by linear
  projections, and cannot recover the intrinsic structure of a
  non-linearly embedded manifold, such as the Swiss roll, that Isomap or
  LLE are designed to unroll (see `Isomap/doc/Description.md`, Section 1).
- **Variance is not the same objective as task relevance.** The direction
  of maximum variance in $X$ need not align with the direction that best
  separates classes, best predicts a downstream target, or is otherwise
  useful for a specific task; PCA's objective (Section 2) is defined
  entirely in terms of the unlabeled data's own spread, with no reference
  to any external notion of relevance. A component can therefore capture
  variance driven by measurement noise or an irrelevant confound, provided
  that source of variation happens to be large in magnitude, while
  discarding a lower-variance direction that is highly informative for the
  task at hand. This is a structural property of the objective in Section
  2, not a failure mode specific to particular datasets.
- Sensitive to feature scaling, as established in Section 5.


##  Code Reference

| File | Contents |
|---|---|
| `code/pca_scratch.py` | PCA implemented from scratch in NumPy via both the covariance-eigendecomposition route and the SVD route, demonstrating their equivalence (Section 3), with `inverse_transform` for reconstruction error |
| `code/pca_sklearn_demo.py` | `sklearn.decomposition.PCA` applied to the digits dataset: scree plot, 2D visualization, and image reconstruction from a reduced number of components, illustrating the compression/denoising trade-off |