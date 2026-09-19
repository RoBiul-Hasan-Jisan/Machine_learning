"""
Anomaly Detection with scikit-learn — LocalOutlierFactor and DBSCAN noise,
with precision/recall evaluation against injected outliers.

Run:
    python anomaly_detection_sklearn_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score


def main():
    rng = np.random.RandomState(0)
    X_normal, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
    X_outliers = rng.uniform(low=X_normal.min(axis=0) - 3, high=X_normal.max(axis=0) + 3,
                              size=(15, 2))
    X = np.vstack([X_normal, X_outliers])
    y_true = np.array([0] * len(X_normal) + [1] * len(X_outliers))  # 1 = anomaly

    lof = LocalOutlierFactor(n_neighbors=20)
    lof_pred = lof.fit_predict(X)          # -1 = outlier, 1 = inlier
    lof_flag = (lof_pred == -1).astype(int)

    dbscan = DBSCAN(eps=0.8, min_samples=5).fit(X)
    db_flag = (dbscan.labels_ == -1).astype(int)

    for name, flag in [("LOF", lof_flag), ("DBSCAN", db_flag)]:
        p = precision_score(y_true, flag, zero_division=0)
        r = recall_score(y_true, flag, zero_division=0)
        print(f"{name}: precision={p:.2f}, recall={r:.2f}, flagged={flag.sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, flag) in zip(axes, [("LOF", lof_flag), ("DBSCAN", db_flag)]):
        ax.scatter(X[:, 0], X[:, 1], c="tab:blue", s=15, label="normal")
        ax.scatter(X[flag == 1, 0], X[flag == 1, 1], c="red", s=40,
                    marker="x", label="flagged anomaly")
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig("anomaly_detection_demo.png", dpi=140)
    print("Saved plot to anomaly_detection_demo.png")


if __name__ == "__main__":
    main()
