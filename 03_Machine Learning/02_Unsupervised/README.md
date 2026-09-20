# Unsupervised Machine Learning

Each topic folder contains:

- `doc/Description.md` — theory: intuition, math, algorithm steps, complexity,
  strengths/weaknesses, and a basic→pro roadmap.
- `code/*_scratch.py` — the algorithm implemented from scratch (NumPy /
  pure Python), independently tested.
- `code/*_sklearn_demo.py` (or `*_mlxtend_demo.py`) — the same problem
  solved with the standard library, for comparison/validation.



### Clustering

1. `01_Clustering` — overview of the field
2. `02_K-Means` — basic, centroid-based
3. `03_Hierarchical_Clustering` — basic, hierarchical, no need to choose
   the number of clusters upfront
4. `04_DBSCAN` — intermediate, density-based, arbitrary-shaped clusters
5. `05_Mean-Shift_Clustering` — intermediate, density-based, mode-seeking
6. `06_Spectral_Clustering` — advanced, graph-based
7. `07_Anomaly_Detection_with_Clustering` — applied use of clustering
   for identifying unusual observations

### Dimensionality Reduction

8. `08_Dimensionality_Reduction` — overview of the field
9. `09_Principal_Component_Analysis` — basic, linear dimensionality reduction
10. `10_Non-negative_Matrix_Factorization` — linear factorization using
    non-negative components
11. `11_Isomap` — intermediate, non-linear, preserves global
    geodesic distances
12. `12_Locally_Linear_Embedding` — intermediate, non-linear, preserves
    local structure
13. `13_t-SNE` — advanced, non-linear, primarily used for visualization

### Association Rule Mining

14. `14_Apriori_Algorithm` — basic, level-wise candidate generation
    with pruning
15. `15_Eclat_Algorithm` — intermediate, depth-first search with
    vertical data representation
16. `16_FP-Growth_Algorithm` — advanced, tree-based pattern mining
    with compressed transaction representation

### Efficiency

17. `17_Efficient_Tree-based_Algorithms` — KD-trees and Ball-trees
    for efficient nearest-neighbor queries used by several
    algorithms above

## Running the Code

```bash
pip install numpy scipy matplotlib scikit-learn pandas

# mlxtend is optional and is used only by the Apriori/FP-Growth
# demo scripts. They fall back to the from-scratch implementations
# if mlxtend is not installed.
pip install mlxtend

cd "02_K-Means/code"
python kmeans_scratch.py

Every *_scratch.py file has been run and verified.

Several implementations were also cross-checked against independent
baselines or related implementations. For example:

Apriori, Eclat, and FP-Growth were compared on the same dataset.
KD-tree/Ball-tree nearest-neighbor results were compared against
brute-force search.
Standard-library implementations were used to validate the
from-scratch implementations where applicable.

The goal is to understand each algorithm from its mathematical
foundation and implementation details through to practical,
library-based usage.
