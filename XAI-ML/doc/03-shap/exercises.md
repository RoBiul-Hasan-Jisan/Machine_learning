# Exercises — 03 SHAP

1. **Axiom check.** For a single patient in the test set, sum all their SHAP values and add the `expected_value`. Confirm it equals the model's raw prediction for that patient (the Efficiency axiom). Show your code and result.
2. **Global vs local.** Pick a feature that ranks high in the global `summary_plot` but has a near-zero SHAP value for one specific patient. Explain how this is possible.
3. **TreeSHAP vs brute force.** Explain, without writing code, why computing exact Shapley values by brute force is infeasible for the 30-feature dataset used in this repo, and how `TreeExplainer` avoids that cost.
4. **Dependence plot.** Change `feature_of_interest` in the dependence plot cell to a different feature (e.g. `'worst radius'`) and to a different `interaction_index`. Does the interaction pattern change the story compared to `'mean texture'`?
5. **Compare patients.** Explain two different patients (change `patient_idx`) with opposite predictions. Which features flip sign between the two explanations, and does that match your medical intuition about tumor severity?
