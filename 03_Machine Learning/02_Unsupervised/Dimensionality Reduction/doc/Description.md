# Dimensionality Reduction — Overview

## 1. Why reduce dimensions?
High-dimensional data is hard to visualize, slow to compute with, and
suffers from the **curse of dimensionality** (distances become less
meaningful, models need exponentially more data to generalize, more
features often means more noise/redundancy). Dimensionality reduction maps
data from `d` dimensions to `k << d` dimensions while preserving as much
"important" structure as possible.

## 2. Two main goals (often in tension)
- **Compression / denoising** — keep most of the variance/information with
  fewer numbers (e.g. PCA for storage, preprocessing, noise removal).
- **Visualization** — project to 2D/3D specifically to *look* at the data's
  cluster/manifold structure (e.g. t-SNE, UMAP), even at the cost of not
  preserving exact distances.

## 3. Linear vs. non-linear methods
| Type | Assumption | Examples |
|---|---|---|
| Linear | Structure lies in (or near) a linear subspace | PCA, NMF |
| Non-linear (manifold learning) | Structure lies on a curved lower-dimensional manifold embedded in high-D space | Isomap, LLE, t-SNE, Spectral Embedding |

The classic illustration: a "Swiss roll" (a 2D sheet rolled up in 3D) — PCA
(linear) cannot unroll it, but Isomap/LLE (non-linear, manifold-aware) can.

## 4. This repo's coverage (basic → pro)
1. **Basic / linear**: **Principal Component Analysis (PCA)** — project onto
   directions of maximum variance; the foundation everything else is
   compared against.
2. **Basic / linear, non-negativity constrained**: **Non-negative Matrix
   Factorization (NMF)** — like PCA but factors are constrained to be
   non-negative, giving more interpretable, "additive" parts (useful for
   images, text/topics).
3. **Intermediate / non-linear (global)**: **Isomap** — preserves
   *geodesic* (along-the-manifold) distances rather than straight-line
   distances.
4. **Intermediate / non-linear (local)**: **Locally Linear Embedding (LLE)**
   — preserves local neighborhood relationships, reconstructing each point
   as a linear combination of its neighbors.
5. **Advanced / non-linear (visualization-focused)**: **t-SNE** — preserves
   local neighborhood *probabilities*, extremely popular for visualizing
   high-dimensional embeddings (e.g. word/image embeddings) in 2D.
6. Also see **Spectral Clustering** (`../Spectral Clustering`), which uses
   the same "graph Laplacian eigenvectors" machinery as a *clustering*
   method — spectral embedding is really dimensionality reduction used for
   clustering.

## 5. How to choose
- Need an interpretable, invertible linear transform, or you'll feed the
  result into another linear model? → **PCA**.
- Need strictly non-negative, "parts-based" factors (e.g. topics in
  documents, parts of a face)? → **NMF**.
- Need to preserve true manifold distances (e.g. for a physically
  meaningful embedding)? → **Isomap**.
- Need to preserve local neighborhood structure with less distortion at
  small scale? → **LLE**.
- Just want the best possible 2D/3D **picture** of cluster structure in
  your data, and don't care about exact distances? → **t-SNE** (or UMAP,
  not covered here but very similar in spirit and faster).

## 6. Evaluating a reduction
- **Explained variance ratio** (PCA-specific): fraction of original
  variance captured by the top-k components.
- **Reconstruction error**: how well can you map back from the low-D
  representation to the original space (only meaningful for methods with
  an explicit inverse, like PCA/NMF).
- **Trustworthiness / neighborhood preservation**: do the k-nearest
  neighbors in the low-D embedding match the k-nearest neighbors in the
  original space? (`sklearn.manifold.trustworthiness`) — the standard way
  to evaluate manifold-learning / visualization methods like t-SNE where
  there's no reconstruction error to check.

## 7. Code in this folder
- `code/dimensionality_reduction_overview.py` — a runnable comparison of
  PCA, Isomap, LLE and t-SNE embedding the classic Swiss-roll and digits
  datasets down to 2D, side by side, to build visual intuition for how each
  method distorts the data differently.

Dedicated from-scratch implementations + deep-dive theory for each specific
method live in their own folders (`Principal Component Analysis/`,
`Non-negative Matrix Factorization/`, `Isomap/`, `Locally Linear
Embedding/`, `t-SNE/`).
