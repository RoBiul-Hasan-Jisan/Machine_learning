# Spectral Clustering


##  Problem Framing

K-Means and Gaussian Mixture Models cluster points according to their
position in the original feature space, and are consequently limited to
recovering approximately convex, isotropic cluster shapes (see
`K-Means/doc/Description.md`, Section 7). Spectral clustering instead
represents the data as a weighted graph, in which nodes are data points
and edge weights encode pairwise similarity, and derives an embedding
from the eigenvectors of a matrix associated with that graph — the graph
Laplacian. Because this embedding is constructed from graph connectivity
rather than raw coordinate distance, clustering in the embedded space (by
K-Means, applied as a final step) can recover structure that is
connected but not convex, such as concentric rings or interleaved
spirals.

##  Similarity Graph Construction

The similarity graph is the sole input encoding the data's geometric
structure, and three constructions are standard:

- **$\varepsilon$-neighborhood graph.** Connect two points if their
  distance is less than $\varepsilon$, typically with unweighted (binary)
  edges.
- **$k$-nearest-neighbor graph.** Connect each point to its $k$ nearest
  neighbors, producing a sparse graph.
- **Fully connected graph with a Gaussian (RBF) kernel.** Assign every
  pair of points an edge weight
  $$w_{ij} = \exp\!\left(-\frac{\lVert x_i - x_j \rVert^2}{2\sigma^2}\right)$$
  so that nearby points receive weight close to 1 and distant points
  receive weight close to 0, with $\sigma$ controlling the rate of decay.

##  The Graph Laplacian

Given a symmetric weight matrix $W$ (with $w_{ij} = 0$ for
non-adjacent pairs, or in the fully connected case, the RBF weights above)
and the diagonal degree matrix $D$, where $D_{ii} = \sum_j w_{ij}$, three
Laplacian variants are used:

| Laplacian | Definition |
|---|---|
| Unnormalized | $L = D - W$ |
| Normalized (symmetric) | $L_{\text{sym}} = D^{-1/2} L D^{-1/2}$ |
| Random-walk | $L_{\text{rw}} = D^{-1} L$ |

##  The Connected-Components Eigenvalue Result

**Claim.** The multiplicity of the eigenvalue $0$ of $L$ equals the number
of connected components of the graph.

**Proof sketch.** For any vector $f \in \mathbb{R}^n$,

$$f^\top L f = f^\top (D - W) f = \sum_{i,j} w_{ij} (f_i - f_j)^2 \geq 0$$

establishing that $L$ is positive semi-definite, so all its eigenvalues
are non-negative and $f^\top L f = 0$ if and only if $f_i = f_j$ for every
pair $(i, j)$ with $w_{ij} > 0$ — that is, if and only if $f$ is constant
on each connected component of the graph. The eigenspace associated with
eigenvalue $0$ is therefore spanned by the indicator vectors of the
graph's connected components, and its dimension equals the number of such
components. $\blacksquare$

**Practical consequence.** In a graph that is fully connected, or
connected apart from noise, the theoretical result gives only a single
zero eigenvalue, which is not directly useful for identifying $k > 1$
clusters. In practice, spectral clustering instead examines the $k$
smallest eigenvalues and their eigenvectors: these correspond to
"almost-separate" components — subsets of the graph that are only weakly
connected to the rest, so that their associated eigenvectors are nearly,
though not exactly, constant on each such subset, by continuity of the
eigenvalue problem in the graph's weights.

##  Algorithm Specification

**Input:** dataset $X$, similarity graph construction method, number of
clusters $k$.

1. Construct the similarity graph and weight matrix $W$ as in Section 2.
2. Compute the (typically normalized) graph Laplacian $L$ as in Section 3.
3. Compute the $k$ eigenvectors of $L$ corresponding to its $k$ smallest
   eigenvalues, and stack them as columns to form an $n \times k$
   embedding matrix $U$; each row of $U$ is the new, low-dimensional
   representation of the corresponding original point.
4. Optionally, normalize each row of $U$ to unit length — a standard step
   when using the normalized Laplacian $L_{\text{sym}}$, since this
   normalization corresponds to the theoretical derivation of the
   normalized-cut relaxation (Section 7).
5. Apply K-Means to the rows of $U$ to obtain the final cluster
   assignments.

The method is named "spectral" because the entire procedure operates on
the spectrum — the eigenvalues and eigenvectors — of the Laplacian matrix,
rather than on the data's original coordinates.

##  Why the Eigenvectors Encode Cluster Structure

The quadratic form derived in Section 4,
$f^\top L f = \sum_{i,j} w_{ij}(f_i - f_j)^2$, provides a direct
interpretation of what the low-eigenvalue eigenvectors represent: this sum
penalizes a function $f$ over the graph's nodes for varying between
strongly connected (high-weight) pairs of nodes, while imposing little
penalty for varying between weakly connected pairs. Because eigenvectors
of $L$ with small eigenvalue $\lambda$ satisfy $f^\top L f = \lambda$ for
unit-norm $f$, these eigenvectors are precisely the smoothest functions
achievable on the graph subject to orthogonality with previously found
eigenvectors: they vary slowly within a well-connected community (since
doing otherwise would incur a large penalty from the many high-weight
edges within that community) and are free to vary sharply between
communities that are only weakly connected to each other. Clustering on
these eigenvectors therefore groups points according to which
well-connected community they belong to, rather than according to their
proximity in the original coordinate space.

##  Connection to Normalized Cut

The behavior described in Section 6 is not merely an analogy: spectral
clustering with the normalized Laplacian $L_{\text{sym}}$ is a continuous
relaxation of the normalized cut (Ncut) graph-partitioning problem. The
normalized cut for a partition of the graph into sets $A$ and its
complement $\bar{A}$ is defined as:

$$\text{Ncut}(A, \bar{A}) = \frac{\text{cut}(A, \bar{A})}{\text{vol}(A)} + \frac{\text{cut}(A, \bar{A})}{\text{vol}(\bar{A})}$$

where $\text{cut}(A, \bar{A}) = \sum_{i \in A, j \in \bar{A}} w_{ij}$ is
the total weight of edges crossing the partition, and $\text{vol}(A)$ is
the sum of degrees of nodes in $A$. Minimizing Ncut exactly over discrete
partitions is NP-hard. Shi and Malik (2000) show that relaxing the
discrete partition indicator to a real-valued vector transforms this
minimization into a generalized eigenvalue problem involving $L$ and $D$,
whose solution is given by the eigenvector of $L_{\text{sym}}$ associated
with the second-smallest eigenvalue (the smallest being the trivial
constant eigenvector, per Section 4). Spectral clustering's use of the
$k$ smallest eigenvectors is the direct generalization of this result to
$k$-way partitioning, which is the formal justification for treating
spectral clustering as an approximate solver for normalized-cut graph
partitioning rather than as a heuristic without a defined combinatorial
objective.

##  Complexity Analysis

| Step | Complexity |
|---|---|
| Full similarity matrix construction | $O(n^2)$ |
| Eigendecomposition (naive, dense) | $O(n^3)$ |
| Eigendecomposition (sparse graph, e.g. $k$-NN, sparse eigensolver such as ARPACK) | Substantially faster in practice than the dense case |

The eigendecomposition step is the primary scalability bottleneck. Using
a sparse similarity graph (Section 2) rather than a fully connected one
is the standard mitigation, since it both reduces memory usage and allows
sparse eigensolvers to be applied.

##  Strengths and Limitations

**Strengths.**
- Recovers non-convex cluster structure that K-Means and GMM cannot,
  since cluster membership is determined by graph connectivity rather
  than distance to a single representative point (Section 6).
- Operates purely on a similarity or affinity matrix, requiring no raw
  coordinate representation of the data — a property that makes it
  directly applicable to graph-structured data, such as social networks,
  where no natural coordinate embedding exists.
- Has a well-defined combinatorial interpretation as an approximate
  solver for normalized-cut graph partitioning (Section 7), rather than
  being justified purely by empirical performance.

**Limitations.**
- Expensive for large $n$, owing to the eigendecomposition cost
  identified in Section 8.
- The number of clusters $k$ must still be chosen, since it determines
  both how many eigenvectors are retained in Step 3 of Section 5 and the
  parameter passed to the final K-Means step.
- Sensitive to the choice of similarity graph and its parameters —
  $\varepsilon$, $k$ in a $k$-NN graph, or $\sigma$ in an RBF kernel —
  since the entire method's input is this graph, and a poorly chosen graph
  can produce a Laplacian whose low eigenvectors do not correspond to any
  meaningful community structure.


##  Code Reference

| File | Contents |
|---|---|
| `code/spectral_scratch.py` | Spectral clustering implemented from scratch in NumPy: RBF similarity graph, normalized Laplacian, eigendecomposition, and K-Means applied to the resulting embedding, reusing the from-scratch K-Means implementation |
| `code/spectral_sklearn_demo.py` | `sklearn.cluster.SpectralClustering` applied to concentric circles, compared against K-Means to illustrate the non-convex-cluster advantage, with an eigenvalue-gap plot |