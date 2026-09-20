"""
Tree-based nearest-neighbor search efficiency — benchmarks brute-force vs
KD-tree vs Ball-tree (via sklearn) as n and dimensionality grow, showing
where tree indexing helps and where the curse of dimensionality erases
the advantage.

Run:
    python tree_efficiency_demo.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def time_query(X, algorithm, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(X)
    t0 = time.perf_counter()
    nn.kneighbors(X[:50])  # query a fixed number of points for a fair comparison
    return time.perf_counter() - t0


def benchmark_vs_n():
    sizes = [500, 1000, 2000, 4000, 8000]
    results = {"brute": [], "kd_tree": [], "ball_tree": []}
    rng = np.random.RandomState(0)
    for n in sizes:
        X = rng.uniform(size=(n, 3))  # low dimension: trees should win clearly
        for algo in results:
            results[algo].append(time_query(X, algo))
    return sizes, results


def benchmark_vs_dimension():
    dims = [2, 5, 10, 20, 40, 80]
    n = 3000
    results = {"brute": [], "kd_tree": [], "ball_tree": []}
    rng = np.random.RandomState(0)
    for d in dims:
        X = rng.uniform(size=(n, d))
        for algo in results:
            results[algo].append(time_query(X, algo))
    return dims, results


def main():
    sizes, results_n = benchmark_vs_n()
    dims, results_d = benchmark_vs_dimension()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for algo, times in results_n.items():
        axes[0].plot(sizes, times, marker="o", label=algo)
    axes[0].set_xlabel("n (dataset size, d=3)")
    axes[0].set_ylabel("query time (s)")
    axes[0].set_title("Scaling with n (fixed low dimension)")
    axes[0].legend()

    for algo, times in results_d.items():
        axes[1].plot(dims, times, marker="o", label=algo)
    axes[1].set_xlabel("dimensionality (fixed n=3000)")
    axes[1].set_ylabel("query time (s)")
    axes[1].set_title("Curse of dimensionality: tree advantage shrinks")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("tree_efficiency_benchmark.png", dpi=140)
    print("Saved benchmark plots to tree_efficiency_benchmark.png")

    print("\nQuery times vs n (d=3):")
    for algo, times in results_n.items():
        print(f"  {algo}: {[round(t, 4) for t in times]}")

    print("\nQuery times vs dimension (n=3000):")
    for algo, times in results_d.items():
        print(f"  {algo}: {[round(t, 4) for t in times]}")


if __name__ == "__main__":
    main()
