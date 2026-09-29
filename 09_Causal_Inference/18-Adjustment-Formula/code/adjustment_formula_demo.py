""" Adjustment formula: exact from a table, then confirmed by simulation."""
import numpy as np

pz = {0: 0.6, 1: 0.4}
pt1_given_z = {0: 0.2, 1: 0.7}
py1 = {(0, 0): 0.10, (0, 1): 0.40, (1, 0): 0.30, (1, 1): 0.60}   # (t, z) -> P(Y=1)

def adjusted(t):
    return sum(py1[(t, z)] * pz[z] for z in (0, 1))

def naive(t):
    pt = {z: (pt1_given_z[z] if t == 1 else 1 - pt1_given_z[z]) for z in (0, 1)}
    p_t = sum(pt[z] * pz[z] for z in (0, 1))
    return sum(py1[(t, z)] * pt[z] * pz[z] / p_t for z in (0, 1))

print("EXACT (from the table)")
print(f"  Adjusted ATE  = {adjusted(1):.3f} - {adjusted(0):.3f} = {adjusted(1)-adjusted(0):.3f}")
print(f"  Naive  ATE    = {naive(1):.3f} - {naive(0):.3f} = {naive(1)-naive(0):.3f}")

rng = np.random.default_rng(0); n = 1_000_000
Z = rng.binomial(1, 0.4, n)
T = rng.binomial(1, np.where(Z == 1, 0.7, 0.2))
Y = rng.binomial(1, np.array([py1[(t, z)] for t, z in [(0,0),(0,1),(1,0),(1,1)]])[2*T+Z])
naive_sim = Y[T == 1].mean() - Y[T == 0].mean()
adj_sim = sum((Z == z).mean() * (Y[(T == 1) & (Z == z)].mean() - Y[(T == 0) & (Z == z)].mean()) for z in (0, 1))
print("\nSIMULATION (n=1,000,000)")
print(f"  Naive    = {naive_sim:.3f}")
print(f"  Adjusted = {adj_sim:.3f}   (truth 0.200)")
