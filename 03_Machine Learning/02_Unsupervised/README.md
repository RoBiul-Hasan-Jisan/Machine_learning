# Unsupervised Machine Learning — From Basic to Pro

Each topic folder contains:
- `doc/Description.md` — theory: intuition, math, algorithm steps, complexity,
  strengths/weaknesses, and a basic→pro roadmap.
- `code/*_scratch.py` — the algorithm implemented from scratch (NumPy /
  pure Python), independently tested.
- `code/*_sklearn_demo.py` (or `*_mlxtend_demo.py`) — the same problem
  solved with the standard library, for comparison/validation.

## Suggested learning order

**Clustering**
1. `01_Clustering` — overview of the field
2. `K-Means` — basic, centroid-based
3. `Hierarchical Clustering` — basic, no need to pick k upfront
4. `DBSCAN` — intermediate, density-based, arbitrary shapes
5. `Mean-Shift Clustering` — intermediate, density-based, mode-seeking
6. `Spectral Clustering` — advanced, graph-based
7. `Anomaly Detection with Clustering` — applied

**Dimensionality Reduction**
8. `Dimensionality Reduction` — overview of the field
9. `Principal Component Analysis` — basic, linear
10. `Non-negative Matrix Factorization` — basic, linear, non-negative parts
11. `Isomap` — intermediate, non-linear (global geodesic distances)
12. `Locally Linear Embedding` — intermediate, non-linear (local structure)
13. `t-SNE` — advanced, non-linear (visualization-focused)

**Association Rule Mining**
14. `Apriori Algorithm` — basic, level-wise with pruning
15. `Eclat Algorithm` — intermediate, depth-first, vertical data format
16. `FP-Growth Algorithm` — advanced, tree-compressed, 2 database scans

**Efficiency**
17. `Efficient Tree-based Algorithms` — KD-trees/Ball-trees that make
    nearest-neighbor queries fast for many of the algorithms above

## Running the code
```bash
pip install numpy scipy matplotlib scikit-learn pandas
# mlxtend is optional (used only in the Apriori/FP-Growth demo scripts,
# which fall back to the from-scratch implementations if it's missing):
pip install mlxtend

cd "K-Means/code" && python kmeans_scratch.py
```
Every `*_scratch.py` file has been run and verified; several were
cross-checked against each other (Apriori == Eclat == FP-Growth on the
same data) or against a brute-force baseline (KD-tree vs brute force) to
confirm correctness.
