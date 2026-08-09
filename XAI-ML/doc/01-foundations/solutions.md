# Solutions — 01 Foundations

1. A black box model is one where we can observe inputs and outputs but cannot understand the internal decision logic (parameters too numerous/complex for human comprehension). Black box: Deep Neural Network. Not a black box: Linear Regression (fully interpretable coefficients).

2. It's "the fundamental law" because it captures the central tension XAI exists to resolve: the models that perform best (deep nets, ensembles) are the hardest to understand, and vice versa. Counterexamples exist at the margins — e.g. a very deep but sparse/monotonic neural network can retain some interpretability; a small decision tree with hundreds of oddly-encoded categorical splits can become surprisingly hard to read despite being "simple" in model class.

3. DARPA framework components: **Prediction** (the model's output), **Explanation** (why it produced that output), **Uncertainty** (how confident the model is), **Counterfactuals** (what would change the outcome). For a loan model: Prediction = "Denied"; Explanation = "High debt-to-income ratio drove this decision"; Uncertainty = "85% confidence"; Counterfactual = "If debt-to-income dropped below 30%, this would likely be approved."

4. Antecedent methods build interpretability *into* the model before/during training (e.g. shallow decision trees, L1-regularized linear regression). Post-hoc methods explain a model *after* it's trained, without changing it (e.g. SHAP, LIME, Grad-CAM, PDP).

5. (a) Regulatory audit of overall behavior → **global** explanation, e.g. global SHAP feature importance or a global surrogate model. (b) Explaining one applicant's rejection → **local** explanation, e.g. SHAP force plot or LIME for that instance.

6. Answers will vary — the key is correctly identifying whether the system is glass-box/black-box and matching the explanation type to the stakeholder's actual question ("why me specifically" = local; "how does this generally work" = global).
