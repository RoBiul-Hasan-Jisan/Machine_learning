# Exercises — 09 XAI Dashboard & Trust Metrics

1. **Run it.** Run `code.ipynb` and record: global top-5 features, the local explanation for `test_instance`, and the final overall trust score.
2. **What-if.** Call `dashboard.what_if_analysis` with a different `feature_to_change` (try a few indices) and `change_percent` values (try -20 and +50). Which feature, when changed, produces the largest swing in confidence?
3. **Trust score design.** The `trust_score` in `trust_metrics` averages accuracy, consistency, and confidence-calibration gap. Critique this design: is a simple average the right way to combine these three very different signals? Propose an alternative (e.g. weighted average, minimum-of-three, or a different formula).
4. **Break it.** Deliberately make the model less trustworthy — e.g. retrain `rf` with `max_depth=1` (very underfit) or on a tiny subsample of the training data (e.g. `X_train[:20]`) — and rerun `trust_metrics`. Which sub-metric drops the most, and does the overall trust score correctly reflect the degradation?
5. **Extend the dashboard.** Add a new method to `XAIDashboard`, e.g. `compare_two_instances(self, instance_a, instance_b)`, that prints the SHAP-based differences between two patients' explanations side by side. Sketch the method signature and describe what it would print.
