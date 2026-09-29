""" Bounds on the probability of necessity and sufficiency (PNS)."""
import numpy as np

rng = np.random.default_rng(0)
n = 3_000_000

def run(p_types, p_treat_by_type, label):
    types = rng.choice(4, size=n, p=p_types)                # 0 always, 1 never, 2 helped, 3 harmed
    Y0 = np.isin(types, [0, 3]).astype(int)
    Y1 = np.isin(types, [0, 2]).astype(int)
    T = rng.binomial(1, np.array(p_treat_by_type)[types])
    Y = np.where(T == 1, Y1, Y0)

    true_pns = np.mean((Y1 == 1) & (Y0 == 0))
    p1, p0 = Y1.mean(), Y0.mean()                           # what an RCT would identify
    pY = Y.mean()
    p_t1y1, p_t0y0 = np.mean((T == 1) & (Y == 1)), np.mean((T == 0) & (Y == 0))
    p_t1y0, p_t0y1 = np.mean((T == 1) & (Y == 0)), np.mean((T == 0) & (Y == 1))

    exp_lo, exp_hi = max(0, p1 - p0), min(p1, 1 - p0)
    obs_lo = max(0, p1 - p0, pY - p0, p1 - pY)
    obs_hi = min(p1, 1 - p0, p_t1y1 + p_t0y0, p1 - p0 + p_t1y0 + p_t0y1)
    print(f"\n{label}")
    print(f"  p1 = {p1:.3f}, p0 = {p0:.3f}, risk difference (ATE) = {p1 - p0:.3f}")
    print(f"  TRUE PNS                          = {true_pns:.3f}")
    print(f"  Bounds from experiment only       = [{exp_lo:.3f}, {exp_hi:.3f}]")
    print(f"  Bounds from experiment + obs data = [{obs_lo:.3f}, {obs_hi:.3f}]   contains truth: {obs_lo - 1e-3 <= true_pns <= obs_hi + 1e-3}")
    print(f"  IF we separately ASSUME monotonicity, PNS is claimed to equal p1-p0 = {p1 - p0:.3f}")

# types: always, never, helped, harmed
run([0.2, 0.4, 0.3, 0.1], [0.7, 0.3, 0.6, 0.3], "Some units are harmed (confounded treatment)")
run([0.2, 0.5, 0.3, 0.0], [0.7, 0.3, 0.6, 0.3], "Monotonicity holds (no harmed units): PNS = risk difference")
