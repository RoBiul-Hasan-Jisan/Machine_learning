"""
03. Interventions 

Builds the structural causal model (SCM) matching theory.md's DAG:

    Z -> T,   Z -> Y,   T -> Y

Then compares:
  - P(Y | T=t)         via plain conditioning on observational data
  - P(Y | do(T=t))      via an explicit intervention: regenerate the
                        population with T forced to t for everyone,
                        severing the Z -> T arrow (graph surgery)

TRUE_EFFECT is the actual causal effect of T on Y, set by us so we can
confirm the do(.)-based estimate recovers it, while the naive conditioning
estimate (in the "confounded" setting) does not.


"""

import numpy as np

rng = np.random.default_rng(seed=0)
n = 20_000
TRUE_EFFECT = 2.0  # the real, causal effect of T on Y


def structural_model(force_T=None):
    """
    force_T: None -> T is generated normally (Z -> T edge active, i.e. observational data)
             0 or 1 -> do(T=force_T): Z -> T edge is severed, T is fixed for everyone
    """
    Z = rng.normal(0, 1, size=n)

    if force_T is None:
        # Observational: T depends on Z (this is the confounding edge Z -> T).
        T = (rng.normal(0, 1, size=n) + 1.5 * Z > 0).astype(float)
    else:
        # do(T = force_T): graph surgery -- the Z -> T edge is deleted.
        T = np.full(n, float(force_T))

    # Y depends on both T (the causal effect we care about) and Z (confounder's
    # direct effect on the outcome) -- this edge is untouched by the intervention.
    noise = rng.normal(0, 1, size=n)
    Y = TRUE_EFFECT * T + 3.0 * Z + noise
    return Z, T, Y


# --- 1. Conditioning: P(Y | T=1) - P(Y | T=0), from observational data ---
Z_obs, T_obs, Y_obs = structural_model(force_T=None)
conditioning_estimate = Y_obs[T_obs == 1].mean() - Y_obs[T_obs == 0].mean()

# --- 2. Intervening: P(Y | do(T=1)) - P(Y | do(T=0)), via graph surgery ---
_, _, Y_do1 = structural_model(force_T=1)
_, _, Y_do0 = structural_model(force_T=0)
do_estimate = Y_do1.mean() - Y_do0.mean()

print(f"True causal effect (by construction):                  {TRUE_EFFECT:.3f}")
print(f"Conditioning estimate,  E[Y|T=1] - E[Y|T=0]:            {conditioning_estimate:.3f}")
print(f"do(.) estimate,         E[Y|do(T=1)] - E[Y|do(T=0)]:    {do_estimate:.3f}")
print()
print("The do(.) estimate matches the true effect closely -- forcing T removes")
print("the Z -> T confounding edge. The conditioning estimate is inflated by")
print("the backdoor path T <- Z -> Y, exactly as theory.md section 2.2 describes.")
