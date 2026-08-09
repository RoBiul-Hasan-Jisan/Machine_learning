# Exercises — 04 LIME

1. **Stability test.** Run the LIME explanation for `patient_idx = 5` three times in a row. Do the top features and weights stay identical? What does this tell you about LIME's reproducibility compared to SHAP?
2. **Kernel width.** LIME's `LimeTabularExplainer` has a `kernel_width` parameter (not set explicitly in this notebook, so it uses a default). Look up what the default is, then re-run with a much smaller and much larger kernel width. How do the explanations change?
3. **num_features.** Change `num_features=10` to `num_features=3` and to `num_features=20`. Does the ranking of the top 3 features stay consistent across these settings?
4. **Compare to SHAP.** For the same `patient_idx`, compare LIME's top 5 features (from this notebook) to SHAP's top 5 features (from `03-shap`). Do they broadly agree? Where do they diverge, and why might that be?
5. **When to prefer LIME.** Describe a real deployment scenario where LIME's speed advantage over SHAP would matter more than SHAP's theoretical guarantees.
