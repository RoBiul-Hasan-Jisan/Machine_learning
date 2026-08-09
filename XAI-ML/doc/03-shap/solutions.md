# Solutions — 03 SHAP

1. `explainer.expected_value[1] + shap_values[idx, :, 1].sum()` should equal `rf.predict_proba(X_test.iloc[[idx]])[0][1]` (up to floating point precision). This is exactly the Efficiency axiom: `Σφᵢ = f(x) − E[f(X)]`, rearranged as `f(x) = E[f(X)] + Σφᵢ`.

2. A feature can have high *global* average impact (large mean |SHAP| across all patients) while having a near-zero effect for one specific patient — this happens when the feature matters a lot in general but this particular patient's value for it happens to sit right at the "neutral" point (e.g. close to the population mean, so it barely shifts the prediction away from baseline for them specifically).

3. Brute-force Shapley values require evaluating the model on all `2^30` (~1 billion) feature subsets per prediction — computationally infeasible. `TreeExplainer` exploits the tree structure itself: it can compute exact Shapley values in time polynomial in the number of trees and leaves (roughly `O(TLD²)` for T trees, L leaves, D depth) by tracking how each possible feature subset routes through the tree's decision paths, without literally enumerating every subset.

4. Answers vary by feature choice, but generally: features closely correlated with tumor size (`worst radius`, `worst perimeter`, `worst area`) tend to show similar monotonic SHAP patterns and strong interaction with each other (since they're geometrically related), while a feature like `mean smoothness` shows a noisier, weaker relationship.

5. Expect the sign of dominant features (e.g. `worst concave points`, `worst area`) to flip between a Malignant-predicted and Benign-predicted patient, consistent with larger/more irregular tumor measurements pushing toward Malignant and smaller/more regular measurements pushing toward Benign — matching clinical intuition that tumor size/shape irregularity signals malignancy.
