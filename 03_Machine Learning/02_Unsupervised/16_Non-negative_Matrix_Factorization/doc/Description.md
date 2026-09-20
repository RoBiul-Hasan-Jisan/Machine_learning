# Non-negative Matrix Factorization (NMF)

##  Problem Definition

Given a non-negative data matrix $X \in \mathbb{R}_{\geq 0}^{n \times d}$
(n samples, d features, all entries non-negative), NMF seeks matrices
$W \in \mathbb{R}_{\geq 0}^{n \times k}$ and $H \in \mathbb{R}_{\geq 0}^{k
\times d}$, both entrywise non-negative, such that:

$$X \approx WH$$

Each row of $H$ is interpreted as a basis component ("part" or "topic")
defined over the original $d$-dimensional feature space, and each row of
$W$ gives the non-negative coefficients specifying how much of each
component is present in the corresponding sample. Because both factors are
constrained to be non-negative, every reconstructed sample is expressed as
a purely additive combination of components — subtraction between
components is not representable under this factorization.

##  Why Non-Negativity Produces Parts-Based Components

The non-negativity constraint is the direct cause of NMF's characteristic
"parts-based" decompositions, and this can be seen by contrast with PCA.

**PCA's components are unconstrained in sign.** PCA's components are
orthogonal directions of maximum variance, with no restriction on the sign
of their entries. Because a sample's reconstruction is a linear
combination of components with both positive and negative coefficients,
individual PCA components generally do not correspond to physically
meaningful, locally interpretable structure on their own — they are
interpretable only in combination, since negative contributions from one
component routinely cancel positive contributions from another.

**Non-negativity forecloses cancellation.** Under NMF's constraint,
because both $W$ and $H$ are entrywise non-negative, no component can
partially cancel another in the reconstruction $WH$: every component can
only add to the reconstructed value, never subtract from it. When applied
to naturally additive, non-negative data — such as pixel intensities,
where a face is physically composed of a fixed arrangement of light-
reflecting parts — this constraint tends to force the discovered
components to correspond to spatially localized, additive structure. This
is the theoretical basis for the empirical result reported by Lee and
Seung (1999): NMF applied to a face-image dataset discovers components
resembling localized facial parts (eyes, nose, mouth regions), while PCA
on the same data produces global "eigenfaces" with mixed-sign pixel
values that are not individually interpretable as parts of a face.

##  The Optimization Problem

NMF minimizes reconstruction error subject to the non-negativity
constraint:

$$\min_{W, H \geq 0} \lVert X - WH \rVert_F^2$$

where $\lVert \cdot \rVert_F$ denotes the Frobenius norm. An alternative
objective based on generalized Kullback–Leibler divergence (equivalently,
a Poisson likelihood) is commonly substituted when $X$ represents count
data, such as word counts in a document-term matrix, since the Frobenius
norm implicitly assumes Gaussian-distributed reconstruction error, which
is a poor model for count data.

**Non-convexity.** This optimization problem is non-convex jointly in $W$
and $H$, even though it is convex in each individually with the other
held fixed. Consequently, unlike PCA, NMF has no closed-form solution:
different initializations of $W$ and $H$ can converge to different local
optima, and the factorization $WH$ is not unique — a further consequence
of non-convexity is that $W$ and $H$ can be rescaled inversely (e.g.,
$W \to WD$, $H \to D^{-1}H$ for any positive diagonal $D$) without
changing $WH$, so the factors themselves are only determined up to this
class of transformations.

##  Multiplicative Update Algorithm

The standard algorithm, due to Lee and Seung, initializes $W$ and $H$
randomly with non-negative entries and alternates the following updates:

$$H \leftarrow H \odot \frac{W^\top X}{W^\top W H + \epsilon}, \qquad
W \leftarrow W \odot \frac{X H^\top}{W H H^\top + \epsilon}$$

where $\odot$ denotes elementwise multiplication, division is elementwise,
and $\epsilon$ is a small positive constant added to avoid division by
zero.

**Why this update rule works.** Two properties make this algorithm
correct and elegant simultaneously:

1. **Non-negativity is preserved automatically.** Because $W$ and $H$ are
   initialized non-negative, and each update multiplies the current value
   by a ratio of non-negative quantities, every subsequent iterate remains
   entrywise non-negative without requiring an explicit projection step.
2. **Monotonic decrease in reconstruction error.** Lee and Seung prove
   that each update in this rule is a special case of a majorize-minimize
   step for the Frobenius-norm objective in Section 3: the multiplicative
   factor is constructed such that the update can never increase
   $\lVert X - WH \rVert_F^2$. This guarantees convergence of the
   objective value, though, consistent with Section 3's non-convexity, only
   to a local minimum determined by the initialization.

##  Complexity Analysis

Each iteration is dominated by the matrix multiplications in the update
rule, costing approximately $O(n \cdot d \cdot k)$ per update — comparable
in order to a single iteration of K-Means (see
`K-Means/doc/Description.md`, Section 6). The algorithm is run for
sufficiently many iterations that the change in reconstruction error
between successive iterations falls below a chosen tolerance.

## Strengths and Limitations

**Strengths.**
- Produces interpretable, additive, parts-based components, as
  established in Section 2 — well suited to topic modeling in text, image
  part decomposition, and audio spectrogram decomposition for source
  separation.
- Naturally matched to data that is inherently sparse and non-negative,
  such as word counts, pixel intensities, and gene expression levels,
  since the constraint in Section 1 aligns with the data's own structure
  rather than being imposed artificially.

**Limitations.**
- Applicable only to non-negative data; data containing negative values
  must be shifted or otherwise transformed into a non-negative range,
  which can be awkward and can distort the interpretability the method is
  otherwise valued for.
- Non-convex (Section 3): sensitive to initialization, with no unique or
  globally optimal solution, in contrast to PCA's uniquely determined,
  ordered, orthogonal components.
- No orthogonality constraint is imposed on the components in $H$; unlike
  PCA's components, NMF's components can overlap or correlate with one
  another, which can complicate interpretation when components are not
  cleanly separated.

##  Relation to Other Methods

NMF and PCA are both linear factorization methods for dimensionality
reduction, differing specifically in the constraint imposed on the
factorization rather than in the general form of the problem: PCA
constrains its components to be orthogonal and orders them by the
variance they capture, while NMF constrains both factors to be
entrywise non-negative and imposes no orthogonality or variance-ranking
structure. This difference in constraint, rather than any difference in
the underlying linear model, is responsible for the qualitative
difference in interpretability discussed in Section 2.

**Application to topic modeling.** NMF is widely used for topic modeling
in natural language processing: given a non-negative document-term
matrix $X$ (e.g., word counts or TF-IDF weights), each row of $H$
corresponds to a topic expressed as a non-negative distribution over
words, and each row of $W$ expresses a document as a non-negative mixture
of topics. This makes NMF a simpler, deterministic-optimization
alternative to Latent Dirichlet Allocation (LDA), which instead models
topics and topic mixtures as draws from Dirichlet-distributed priors
within a fully generative probabilistic model.


##  Code Reference

| File | Contents |
|---|---|
| `code/nmf_scratch.py` | NMF implemented from scratch in NumPy using multiplicative update rules, with reconstruction-error tracking |
| `code/nmf_sklearn_demo.py` | `sklearn.decomposition.NMF` applied to topic modeling on a small text corpus (printing top words per discovered topic) and to face-image parts decomposition, compared against PCA |