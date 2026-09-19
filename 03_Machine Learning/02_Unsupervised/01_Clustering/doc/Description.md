# Clustering — Overview

## 1. What is Clustering?
Clustering is **unsupervised learning**: given data with *no labels*, group points so that
points in the same group (cluster) are more similar to each other than to points in other groups.

There is no ground truth "correct" grouping — success is judged by internal structure
(compactness, separation) or, if labels exist only for evaluation, by external metrics.

## 2. Why it matters (use cases)
- Customer segmentation (marketing)
- Anomaly / fraud detection (outliers = far from any cluster)
- Document / topic grouping
- Image compression (color quantization)
- Biology: gene expression grouping
- Preprocessing step before supervised learning

## 3. Families of clustering algorithms
| Family | Idea | Examples |
|---|---|---|
| Centroid-based | Minimize distance to a cluster center | K-Means, Mean-Shift |
| Density-based | Clusters = dense regions separated by sparse regions | DBSCAN |
| Hierarchical | Build a tree of nested clusters | Agglomerative / Divisive |
| Graph-based | Use similarity graph + spectral properties | Spectral Clustering |
| Distribution-based | Assume a generative probability model | Gaussian Mixture Models |

Each has its own folder in this repo with theory + from-scratch code.

## 4. Distance / similarity — the basic building block
Most algorithms need a notion of "how similar are two points":
- Euclidean distance: `d(x,y) = sqrt(sum((x_i - y_i)^2))`
- Manhattan distance: `sum(|x_i - y_i|)`
- Cosine similarity: `dot(x,y) / (||x|| * ||y||)` (good for text/high-dim sparse data)

## 5. How do you know clustering worked? (Evaluation)
**Without labels (internal):**
- **Inertia / WCSS** (within-cluster sum of squares) — lower is tighter clusters (used by K-Means)
- **Silhouette score** ∈ [-1, 1]: for each point, `(b - a) / max(a, b)` where `a` = mean
  intra-cluster distance, `b` = mean distance to nearest other cluster. Closer to 1 is better.
- **Davies–Bouldin index** — lower is better (ratio of intra-cluster to inter-cluster distance).

**With labels available only for evaluation (external):**
- Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Homogeneity/Completeness.

## 6. Basic → Pro roadmap in this repo
1. **Basic**: K-Means (centroid-based, assumes round/convex clusters, must pick k)
2. **Basic**: Hierarchical Clustering (no need to pick k upfront, gives a dendrogram)
3. **Intermediate**: DBSCAN (density-based, finds arbitrary shapes, detects noise/outliers)
4. **Intermediate**: Mean-Shift (density-based, no need to pick k, mode-seeking)
5. **Advanced**: Spectral Clustering (graph-based, handles non-convex clusters via
   eigen-decomposition of a graph Laplacian)
6. **Applied**: Anomaly Detection with Clustering (using cluster distance/density as an
   outlier score)

## 7. Practical tips
- Always **scale/standardize features** before distance-based clustering (StandardScaler).
- Try multiple algorithms — no single method is best for all shapes.
- Use the elbow method / silhouette analysis to pick `k` when a method requires it.
- Visualize with PCA/t-SNE (see `Dimensionality Reduction`) when data has >2 dimensions.

See `code/clustering_overview.py` for a runnable side-by-side comparison of the main
algorithm families on the same synthetic datasets (blobs, moons, anisotropic blobs).
