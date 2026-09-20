"""
Agglomerative Hierarchical Clustering from scratch (NumPy only).

Implements single, complete, average, and Ward linkage, builds the full
merge history (usable to draw a dendrogram), and provides cut_tree(k) to
get a flat clustering with k clusters.

Run:
    python hierarchical_scratch.py
"""
import numpy as np


class AgglomerativeScratch:
    def __init__(self, linkage="ward"):
        assert linkage in ("single", "complete", "average", "ward")
        self.linkage = linkage
        self.merges_ = None   # list of (cluster_a, cluster_b, distance, new_cluster_id)
        self.labels_ = None

    def _pairwise_sq_dists(self, X):
        diff = X[:, None, :] - X[None, :, :]
        return np.sum(diff ** 2, axis=2)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # cluster_points[cid] = list of original point indices in that cluster
        cluster_points = {i: [i] for i in range(n)}
        active = set(range(n))
        next_id = n

        sq_dists = self._pairwise_sq_dists(X)

        def cluster_distance(ca, cb):
            pts_a, pts_b = cluster_points[ca], cluster_points[cb]
            d = sq_dists[np.ix_(pts_a, pts_b)] ** 0.5
            if self.linkage == "single":
                return d.min()
            if self.linkage == "complete":
                return d.max()
            if self.linkage == "average":
                return d.mean()
            if self.linkage == "ward":
                # increase in within-cluster SSE if we merge ca and cb
                pts = pts_a + pts_b
                merged_mean = X[pts].mean(axis=0)
                sse_merged = np.sum((X[pts] - merged_mean) ** 2)
                mean_a, mean_b = X[pts_a].mean(axis=0), X[pts_b].mean(axis=0)
                sse_a = np.sum((X[pts_a] - mean_a) ** 2)
                sse_b = np.sum((X[pts_b] - mean_b) ** 2)
                return sse_merged - sse_a - sse_b

        merges = []
        while len(active) > 1:
            best = None
            active_list = list(active)
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    ca, cb = active_list[i], active_list[j]
                    d = cluster_distance(ca, cb)
                    if best is None or d < best[0]:
                        best = (d, ca, cb)

            dist, ca, cb = best
            cluster_points[next_id] = cluster_points[ca] + cluster_points[cb]
            active.discard(ca)
            active.discard(cb)
            active.add(next_id)
            merges.append((ca, cb, dist, next_id))
            next_id += 1

        self.merges_ = merges
        self.n_samples_ = n
        self._cluster_points_final = cluster_points
        return self

    def cut_tree(self, k):
        """Return flat cluster labels (0..k-1) for k clusters, by undoing the
        last (k-1) merges (i.e. stopping the merge process early)."""
        n = self.n_samples_
        cluster_points = {i: [i] for i in range(n)}
        active = set(range(n))
        n_merges_to_apply = n - k
        assert 0 <= n_merges_to_apply <= len(self.merges_), "k out of range"

        for step in range(n_merges_to_apply):
            ca, cb, dist, new_id = self.merges_[step]
            cluster_points[new_id] = cluster_points[ca] + cluster_points[cb]
            active.discard(ca)
            active.discard(cb)
            active.add(new_id)

        labels = np.full(n, -1)
        for label_idx, cid in enumerate(sorted(active)):
            for pt in cluster_points[cid]:
                labels[pt] = label_idx
        self.labels_ = labels
        return labels


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=40, centers=3, cluster_std=0.6, random_state=1)

    model = AgglomerativeScratch(linkage="ward").fit(X)
    labels = model.cut_tree(k=3)
    print("Cluster sizes:", np.bincount(labels))
    print("First 10 merges (cluster_a, cluster_b, distance):")
    for ca, cb, d, new_id in model.merges_[:10]:
        print(f"  merge {ca} + {cb} -> {new_id}  at distance {d:.3f}")
