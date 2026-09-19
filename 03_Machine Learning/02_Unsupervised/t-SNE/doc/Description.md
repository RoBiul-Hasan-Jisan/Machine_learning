# t-SNE (t-Distributed Stochastic Neighbor Embedding)

## 1. Intuition
t-SNE is designed for one specific job: making **great 2D/3D pictures** of
high-dimensional data for visualization, by preserving *local* neighborhood
structure — points that are close in high-D should stay close in the
low-D picture; it does **not** try to preserve distances between far-apart
points or global structure faithfully (unlike PCA or Isomap).

It's the most popular way to visualize things like word embeddings, image
embeddings, or single-cell genomics data in 2D.

## 2. Algorithm — two probability distributions
**Step 1 — pairwise similarities in high-D space.** For every pair of
points `i, j`, define a conditional probability that `i` would "pick" `j`
as its neighbor, using a Gaussian centered at `i`:

```
p_{j|i} = exp(-||x_i - x_j||^2 / 2*sigma_i^2) / Σ_{k != i} exp(-||x_i - x_k||^2 / 2*sigma_i^2)
```

`sigma_i` is chosen *per point* so that the resulting distribution has a
target **perplexity** (roughly: an effective number of neighbors — a
smoothed measure of local neighborhood size). Symmetrize:
`p_ij = (p_{j|i} + p_{i|j}) / (2n)`.

**Step 2 — pairwise similarities in low-D space.** For the low-dimensional
points `y_i`, define similarity using a **Student-t distribution with 1
degree of freedom** (heavier tails than a Gaussian):

```
q_ij = (1 + ||y_i - y_j||^2)^-1  /  Σ_{k != l} (1 + ||y_k - y_l||^2)^-1
```

**Step 3 — minimize the mismatch.** Move the `y_i` points to minimize the
**KL divergence** between the two distributions `P` and `Q`:

```
KL(P || Q) = Σ_ij p_ij * log(p_ij / q_ij)
```

via gradient descent.

## 3. Why a Student-t (heavy-tailed) distribution in low-D?
This is the "t" in t-SNE, and it solves the **"crowding problem"**: in
high dimensions there's a lot of room, so many moderately-distant points
can all be moderately close to a given point. Squeezing that into 2D with
a Gaussian would force too many points to compete for the same limited
"close" space. The Student-t distribution's heavy tails let
moderately-distant points in the low-D map stay a bit *more spread out*
without heavily penalizing the loss — giving t-SNE's characteristic
well-separated, visually appealing clusters.

## 4. Perplexity — the key hyperparameter
Perplexity (typically 5–50) controls the effective neighborhood size used
when computing `sigma_i`.
- **Low perplexity** → focuses on very local structure, can fragment
  clusters or reveal fine sub-structure (or noise).
- **High perplexity** → considers a broader neighborhood, tends to merge
  small clusters, smoother global impression.
- There is no single "right" value — it's standard practice to try several
  and compare (van der Maaten & Hinton's original paper suggests it's
  fairly robust in the 5-50 range, but results *do* change).

## 5. Important caveats when reading a t-SNE plot
- **Cluster sizes/densities in the plot are NOT meaningful** — t-SNE can
  expand sparse regions and compress dense ones; don't read "how spread
  out a cluster looks" as "how spread out it really is."
- **Distances between separate clusters are NOT reliable** — two clusters
  being close or far apart in the 2D plot doesn't necessarily reflect their
  true high-D relationship.
- **Different runs (even with the same perplexity) can give different-
  looking layouts** — t-SNE is stochastic (random initialization,
  non-convex optimization) unless you fix the random seed.
- Always try a few perplexity values / seeds before drawing conclusions
  from a single t-SNE plot.

## 6. Complexity
- Naive: `O(n^2)` per gradient-descent iteration (all pairwise
  similarities) — expensive for large n.
- **Barnes-Hut t-SNE** (the default in `sklearn`/most libraries for larger
  n) approximates far-away interactions using a spatial tree, bringing this
  down to roughly `O(n log n)`.

## 7. Strengths / Weaknesses
**Strengths**: outstanding for visualizing cluster structure in
high-dimensional data; the heavy-tailed low-D distribution gives visually
crisp, well-separated clusters.
**Weaknesses**: doesn't preserve global structure or true distances;
stochastic and hyperparameter-sensitive; slow on very large datasets
without the Barnes-Hut approximation; has no simple `transform()` for new
points (it's not a general-purpose learned mapping like PCA — you must
refit on the combined old+new data, or use a different method like UMAP
that supports out-of-sample transform more naturally).

## 8. Basic → Pro
1. **Basic**: fixed perplexity, single 2D embedding on a toy dataset,
   gradient descent from scratch.
2. **Intermediate**: perplexity sweep, multiple random seeds, understand
   the resulting variability.
3. **Pro**: Barnes-Hut acceleration for scalability (library-level, not
   reimplemented from scratch here — see the complexity note above); or
   compare with **UMAP** (a related but faster, differently-motivated
   modern alternative, not covered in this repo but worth knowing about).

## 9. Code in this folder
- `code/tsne_scratch.py` — the core t-SNE math from scratch in NumPy:
  per-point sigma search to hit a target perplexity, symmetrized
  high-D affinities, Student-t low-D affinities, and gradient-descent
  optimization of the KL divergence (on a small dataset, since this naive
  `O(n^2)` version is for learning, not for production-scale data).
- `code/tsne_sklearn_demo.py` — `sklearn.manifold.TSNE` on the digits
  dataset with a perplexity sweep, illustrating how the picture changes.
