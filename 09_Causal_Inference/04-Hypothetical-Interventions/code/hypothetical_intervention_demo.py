"""
04. Hypothetical Interventions 

Uses the same coffee / Z / T / Y structural causal model as Module 03:

    Z = U_Z
    T = f_T(Z, U_T)                (irrelevant here -- we condition on observed T)
    Y = TRUE_EFFECT * T + 3*Z + U_Y

For ONE specific simulated individual, with a hidden/private noise term U_Y:
  1. ABDUCTION: recover U_Y from their observed (Z, T, Y), using an ASSUMED
     structural equation for Y (which may or may not match the true one).
  2. ACTION:    hypothetically flip their treatment to 1 - T.
  3. PREDICTION: compute their counterfactual Y under the flipped treatment.

We then compare the computed counterfactual to the TRUE counterfactual for
that same individual (only knowable because we are the ones simulating the
data and kept their true U_Y around for this comparison).

"""

import numpy as np

rng = np.random.default_rng(seed=0)
TRUE_EFFECT = 2.0  # matches Module 03's intervention_demo.py


def draw_one_individual():
    """Simulate one individual's full data, including their private noise U_Y."""
    Z = rng.normal(0, 1)
    T = float(rng.binomial(1, 0.5))
    U_Y = rng.normal(0, 1)
    Y = TRUE_EFFECT * T + 3.0 * Z + U_Y
    return Z, T, Y, U_Y  # U_Y is only for our own later comparison -- "hidden" from the analysis


def abduction_action_prediction(Z, T_observed, Y_observed, assumed_effect_coef):
    """
    Steps 1-3 from theory.md section 2.3, using an ASSUMED coefficient on T
    (assumed_effect_coef). If assumed_effect_coef == TRUE_EFFECT, the analysis
    is using the correct structural equation; otherwise it is deliberately wrong.
    """
    # Step 1: Abduction -- solve Y = assumed_effect_coef*T + 3*Z + U_Y for U_Y
    U_Y_recovered = Y_observed - assumed_effect_coef * T_observed - 3.0 * Z

    # Step 2: Action -- flip the treatment
    T_counterfactual = 1.0 - T_observed

    # Step 3: Prediction -- propagate forward with the recovered noise
    Y_counterfactual = assumed_effect_coef * T_counterfactual + 3.0 * Z + U_Y_recovered

    return Y_counterfactual


# --- 1 & 2: draw one individual, then run the procedure with the CORRECT assumed model ---
Z, T, Y, U_Y_true = draw_one_individual()
print(f"Observed individual: Z={Z:.3f}, T={T:.0f}, Y={Y:.3f}  (their true U_Y={U_Y_true:.3f}, hidden from the analysis)")

Y_cf_correct = abduction_action_prediction(Z, T, Y, assumed_effect_coef=TRUE_EFFECT)

# The TRUE counterfactual, computable here only because we know U_Y_true ourselves.
T_flipped = 1.0 - T
Y_cf_true = TRUE_EFFECT * T_flipped + 3.0 * Z + U_Y_true

print(f"\n=== Correct assumed model (coefficient on T = {TRUE_EFFECT}) ===")
print(f"Computed counterfactual Y (T flipped to {T_flipped:.0f}): {Y_cf_correct:.3f}")
print(f"True counterfactual Y (using the hidden true U_Y):       {Y_cf_true:.3f}")
print(f"Match: {np.isclose(Y_cf_correct, Y_cf_true)}")

# --- 3. Repeat with a DELIBERATELY WRONG assumed coefficient ---
print("\n=== Deliberately wrong assumed models ===")
for wrong_coef in [1.0, 5.0, 10.0]:
    Y_cf_wrong = abduction_action_prediction(Z, T, Y, assumed_effect_coef=wrong_coef)
    error = Y_cf_wrong - Y_cf_true
    print(f"Assumed coefficient={wrong_coef:>4}: computed counterfactual={Y_cf_wrong:.3f}, "
          f"true counterfactual={Y_cf_true:.3f}, error={error:+.3f}")

print("\nNotice the wrong-model counterfactuals are just as confidently computed as the")
print("correct one -- nothing in the abduction step itself signals a problem. The size")
print("of the error grows with how wrong the assumed coefficient is, exactly as")
print("theory.md section 2.5 warns.")
