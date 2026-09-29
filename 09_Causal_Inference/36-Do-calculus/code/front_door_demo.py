""" Front-door adjustment when U is unmeasured (backdoor impossible)."""
import numpy as np

rng = np.random.default_rng(0)
n = 2_000_000

def gen(n, do_T=None):
    U = rng.binomial(1, 0.5, n)
    T = rng.binomial(1, 0.2 + 0.6 * U) if do_T is None else np.full(n, do_T)
    M = rng.binomial(1, 0.1 + 0.7 * T)                    # M depends only on T
    Y = rng.binomial(1, 0.1 + 0.4 * M + 0.3 * U)          # no direct T -> Y edge
    return T, M, Y

T, M, Y = gen(n)                                          # observational data (U hidden)
truth = gen(n, 1)[2].mean() - gen(n, 0)[2].mean()
naive = Y[T == 1].mean() - Y[T == 0].mean()

def adjust_M():                                           # (wrongly) adjust for M
    return sum((M == m).mean() * (Y[(T == 1) & (M == m)].mean() - Y[(T == 0) & (M == m)].mean()) for m in (0, 1))

def front_door(t):
    total = 0.0
    for m in (0, 1):
        p_m_given_t = (M[T == t] == m).mean()
        inner = sum((T == tp).mean() * Y[(T == tp) & (M == m)].mean() for tp in (0, 1))
        total += p_m_given_t * inner
    return total

fd = front_door(1) - front_door(0)
print(f"True effect (simulate do(T))          = {truth:.3f}   (exact: 0.4*0.7 = 0.280)")
print(f"Naive difference (U unmeasured)       = {naive:.3f}")
print(f"Adjusting for M                       = {adjust_M():.3f}")
print(f"Front-door estimate (observed data)   = {fd:.3f}")
