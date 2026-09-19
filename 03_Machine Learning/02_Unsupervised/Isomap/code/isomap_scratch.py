"""
Isomap from scratch (NumPy + SciPy's sparse shortest-path solver, which is
the standard, efficient way to do all-pairs shortest paths -- reimplementing
Dijkstra by hand for every node in pure Python would be needlessly slow;
the graph construction, geodesic-distance choice, and classical MDS below
are the actual "from scratch" Isomap logic).

Run:
    python isomap_scratch.py
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


class IsomapScratch:
    def __init__(self, n_neighbors=10, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.embedding_ = None

    def _build_knn_graph(self, X):
        n = X.shape[0]
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(X)
        distances, indices = nn.kneighbors(X)

        rows, cols, data = [], [], []
        for i in range(n):
            for j_idx in range(1, self.n_neighbors + 1):  # skip self (index 0)
                j = indices[i, j_idx]
                d = distances[i, j_idx]
                rows.append(i)
                cols.append(j)
                data.append(d)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        # symmetrize: take the min distance if edges disagree
        graph = graph.maximum(graph.T)
        return graph

    def _classical_mds(self, D):
        """Classical MDS: recover a low-dim embedding whose Euclidean
        distances best match the given distance matrix D."""
        n = D.shape[0]
        D_sq = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n  # centering matrix
        B = -0.5 * J @ D_sq @ J              # double-centered matrix

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order][:self.n_components]
        eigenvectors = eigenvectors[:, order][:, :self.n_components]

        eigenvalues_clipped = np.clip(eigenvalues, 0, None)  # numerical safety
        embedding = eigenvectors * np.sqrt(eigenvalues_clipped)
        return embedding

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        graph = self._build_knn_graph(X)
        geodesic_dist = shortest_path(graph, method="D", directed=False)

        if np.isinf(geodesic_dist).any():
            n_disconnected = np.isinf(geodesic_dist).sum()
            print(f"Warning: graph is disconnected ({n_disconnected} unreachable pairs); "
                  f"try increasing n_neighbors.")
            # replace inf with a large finite value so MDS doesn't blow up
            finite_max = geodesic_dist[np.isfinite(geodesic_dist)].max()
            geodesic_dist[np.isinf(geodesic_dist)] = finite_max * 2

        self.embedding_ = self._classical_mds(geodesic_dist)
        return self.embedding_


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll

    X, color = make_swiss_roll(n_samples=500, noise=0.05, random_state=42)

    model = IsomapScratch(n_neighbors=10, n_components=2)
    Z = model.fit_transform(X)

    print("Original shape:", X.shape, "-> Embedded shape:", Z.shape)
    # sanity check: correlation between the manifold's intrinsic "color"
    # coordinate and the first embedding dimension should be high if the
    # roll was successfully "unrolled"
    corr = np.corrcoef(Z[:, 0], color)[0, 1]
    print(f"Correlation between embedding dim 0 and true manifold coordinate: {corr:.3f}")
