""" Adjusting for a noisy proxy of a confounder (e.g. an embedding-derived score)."""
import numpy as np

rng = np.random.default_rng(0)
n = 300_000

Z = rng.normal(0, 1, n)                          # true, unmeasured confounder
T = rng.binomial(1, 1 / (1 + np.exp(-1.2 * Z)))
Y = 2.0 * T + 1.5 * Z + rng.normal(0, 1, n)       # true ATE = 2.0

def adjusted_estimate(proxy, n_bins=10):
    bins = np.digitize(proxy, np.quantile(proxy, np.linspace(0, 1, n_bins + 1)[1:-1]))
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if (T[m] == 1).any() and (T[m] == 0).any():
            total += m.mean() * (Y[m & (T == 1)].mean() - Y[m & (T == 0)].mean())
    return total

naive = Y[T == 1].mean() - Y[T == 0].mean()
perfect = adjusted_estimate(Z)
print(f"True ATE = 2.000")
print(f"Naive (no adjustment)         = {naive:.3f}")
print(f"Adjusted for TRUE Z           = {perfect:.3f}\n")

print(f"{'proxy noise sd':<16}{'corr(proxy, Z)':>16}{'adjusted estimate':>19}{'% of naive bias removed':>26}")
naive_bias = naive - 2.0
for noise_sd in (3.0, 1.5, 1.0, 0.5, 0.1):
    proxy = Z + rng.normal(0, noise_sd, n)                 # "embedding-style" noisy proxy
    est = adjusted_estimate(proxy)
    removed = 1 - (est - 2.0) / naive_bias
    print(f"{noise_sd:<16}{np.corrcoef(proxy, Z)[0, 1]:>16.3f}{est:>19.3f}{removed:>26.1%}")
