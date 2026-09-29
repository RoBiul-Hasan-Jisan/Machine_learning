""" Difference-in-differences with a 6-period panel and an event study."""
import numpy as np

rng = np.random.default_rng(0)
n_per_group, periods, t_start, DELTA = 5_000, 6, 3, 2.0

def simulate(trend_diff):
    rows = []
    for g in (0, 1):                                   # 1 = treated group
        a = rng.normal(5.0 if g else 0.0, 1.0, n_per_group)       # groups differ in LEVEL
        for t in range(periods):
            y = a + 1.0 * t + 0.8 * np.sin(t) + trend_diff * t * g \
                + DELTA * (g == 1 and t >= t_start) + rng.normal(0, 1, n_per_group)
            rows.append((g, t, y))
    return rows

def summarize(rows):
    m = {(g, t): y.mean() for g, t, y in rows}
    gap = {t: m[(1, t)] - m[(0, t)] for t in range(periods)}
    pre = [t for t in range(periods) if t < t_start]
    post = [t for t in range(periods) if t >= t_start]
    did = (np.mean([m[(1, t)] for t in post]) - np.mean([m[(1, t)] for t in pre])) - \
          (np.mean([m[(0, t)] for t in post]) - np.mean([m[(0, t)] for t in pre]))
    naive_cross = np.mean([gap[t] for t in post])
    naive_prepost = np.mean([m[(1, t)] for t in post]) - np.mean([m[(1, t)] for t in pre])
    event = {t: gap[t] - gap[t_start - 1] for t in range(periods)}
    return did, naive_cross, naive_prepost, event

for label, td in (("Parallel trends hold", 0.0), ("Treated group trends 0.3/period faster", 0.3)):
    did, cross, prepost, event = summarize(simulate(td))
    print(f"\n=== {label} (true effect = {DELTA}) ===")
    print(f"  post-period treated - control (cross-section) = {cross:.3f}")
    print(f"  treated before/after only                      = {prepost:.3f}")
    print(f"  Difference-in-differences                      = {did:.3f}")
    print("  event-study coefficients (ref = period 2): " +
          "  ".join(f"t{t}:{event[t]:+.2f}" for t in range(periods)))
