"""
10. DAGs and Probability Distributions 

  1. Simulates the confounder DAG (Z -> T, Z -> Y, T -> Y) by drawing from its
     FACTORIZED form P(z) P(t|z) P(y|t,z), then empirically re-estimates the
     joint P(z,t,y) from samples and confirms it matches the product of the
     empirically estimated conditional/marginal pieces -- verifying the
     factorization theorem numerically rather than just asserting it.

  2. Extends to a 4th, causally isolated variable W and confirms the two
     simplifications predicted by the DAG: P(w|z,t) ~= P(w), and
     P(y|z,t,w) ~= P(y|z,t).

  3. Shows two different (Markov-equivalent) 2-variable DAGs are BOTH
     compatible with the same simulated data, illustrating non-identifiability.

Run with: python factorization_demo.py
"""

import numpy as np

rng = np.random.default_rng(seed=0)
n = 500_000

# --- 1. Confounder DAG: simulate from the factorized form, then verify ---
print("=== 1. Verifying the factorization P(z,t,y) = P(z) P(t|z) P(y|t,z) ===")
Z = rng.binomial(1, 0.5, size=n)
T = rng.binomial(1, np.where(Z == 1, 0.7, 0.3))
Y = rng.binomial(1, np.where((Z == 1) & (T == 1), 0.8,
                    np.where((Z == 1) & (T == 0), 0.6,
                    np.where((Z == 0) & (T == 1), 0.5, 0.2))))

def empirical_joint_prob(z_val, t_val, y_val):
    return np.mean((Z == z_val) & (T == t_val) & (Y == y_val))

def empirical_factorized_prob(z_val, t_val, y_val):
    p_z = np.mean(Z == z_val)
    p_t_given_z = np.mean(T[Z == z_val] == t_val)
    p_y_given_tz = np.mean(Y[(Z == z_val) & (T == t_val)] == y_val)
    return p_z * p_t_given_z * p_y_given_tz

print(f"{'z':>2} {'t':>2} {'y':>2} | {'joint P(z,t,y)':>15} | {'factorized product':>19}")
for z_val in [0, 1]:
    for t_val in [0, 1]:
        for y_val in [0, 1]:
            joint = empirical_joint_prob(z_val, t_val, y_val)
            factorized = empirical_factorized_prob(z_val, t_val, y_val)
            print(f"{z_val:>2} {t_val:>2} {y_val:>2} | {joint:>15.4f} | {factorized:>19.4f}")

# --- 2. Add an isolated variable W and check the DAG's predicted simplifications ---
print("\n=== 2. Isolated variable W: checking P(w|z,t) ~= P(w), P(y|z,t,w) ~= P(y|z,t) ===")
W = rng.binomial(1, 0.4, size=n)  # independent of everything

p_w = np.mean(W == 1)
print(f"P(W=1) overall: {p_w:.4f}")
for z_val in [0, 1]:
    for t_val in [0, 1]:
        mask = (Z == z_val) & (T == t_val)
        p_w_given_zt = np.mean(W[mask] == 1)
        print(f"  P(W=1 | Z={z_val}, T={t_val}) = {p_w_given_zt:.4f}  (should be close to {p_w:.4f})")

print()
for z_val in [0, 1]:
    for t_val in [0, 1]:
        p_y_given_zt = np.mean(Y[(Z == z_val) & (T == t_val)] == 1)
        p_y_given_ztw1 = np.mean(Y[(Z == z_val) & (T == t_val) & (W == 1)] == 1)
        print(f"  P(Y=1|Z={z_val},T={t_val}) = {p_y_given_zt:.4f}   "
              f"P(Y=1|Z={z_val},T={t_val},W=1) = {p_y_given_ztw1:.4f}  (should match)")

# --- 3. Markov equivalence: two different DAGs, same compatible distribution ---
print("\n=== 3. Markov equivalence: A->B and A<-B are both 'compatible' with the same data ===")
A = rng.binomial(1, 0.5, size=n)
B = np.where(A == 1, rng.binomial(1, 0.8, size=n), rng.binomial(1, 0.2, size=n))

# "A -> B" factorization: P(a)P(b|a)
p_a = np.mean(A == 1)
p_b_given_a1 = np.mean(B[A == 1] == 1)
p_b_given_a0 = np.mean(B[A == 0] == 1)
print(f"P(A=1)={p_a:.3f}, P(B=1|A=1)={p_b_given_a1:.3f}, P(B=1|A=0)={p_b_given_a0:.3f}"
      "  -- a valid 'A -> B' factorization")

# "A <- B" factorization: P(b)P(a|b) -- computed from the SAME data
p_b = np.mean(B == 1)
p_a_given_b1 = np.mean(A[B == 1] == 1)
p_a_given_b0 = np.mean(A[B == 0] == 1)
print(f"P(B=1)={p_b:.3f}, P(A=1|B=1)={p_a_given_b1:.3f}, P(A=1|B=0)={p_a_given_b0:.3f}"
      "  -- an EQUALLY valid 'A <- B' factorization")
print("Both factorizations reproduce the same joint distribution -- the data alone")
print("cannot tell you which arrow direction is the true causal one.")
