# Solutions — 09 XAI Dashboard & Trust Metrics

1. Results vary by run, but expect global top features to include `worst area`, `worst concave points`, `mean concave points`; local explanation for the first test instance should list SHAP and LIME feature contributions aligned in direction (mostly); trust score is usually reported > 0.8 ("High trust") for this well-performing Random Forest on a relatively easy dataset.

2. Features related to tumor size/shape (e.g. index corresponding to `worst area` or `worst concave points`) typically produce the largest confidence swings, since the model's decision boundary is most sensitive to these — consistent with what SHAP/PDP already showed in earlier lessons.

3. A simple average can hide a serious failure in one dimension — e.g. a model with perfect accuracy and consistency but terrible calibration (confidently wrong as often as confidently right) could still average out to a "moderate trust" score, masking the calibration problem. An alternative: report each sub-metric separately and flag the model as untrustworthy if *any* metric falls below a threshold (min-of-three logic), rather than averaging, since trust should arguably be limited by the *weakest* link, not the average.

4. Under-training with `max_depth=1` or a tiny subsample sharply reduces accuracy (the model can barely split the data) and typically also worsens confidence calibration (the model becomes falsely confident or uniformly unconfident). Local consistency often stays relatively high (a simple/stump model is very stable to small noise, ironically, because it barely uses most features) — showing that a "high consistency" score alone doesn't imply a good model, reinforcing why trust needs multiple, not single, metrics.

5. Sketch:
```python
def compare_two_instances(self, instance_a, instance_b, label_a="A", label_b="B"):
    """Print SHAP-based feature contributions for two instances side by side."""
    shap_a = self.explainer.shap_values(instance_a.reshape(1, -1))
    shap_b = self.explainer.shap_values(instance_b.reshape(1, -1))
    # ... extract per-class SHAP vectors as in explain_prediction ...
    df = pd.DataFrame({
        'feature': self.feature_names,
        label_a: shap_a_vals,
        label_b: shap_b_vals,
        'difference': shap_a_vals - shap_b_vals
    }).sort_values('difference', key=abs, ascending=False)
    print(df.head(10))
```
This would print, for each feature, its SHAP contribution for both patients and the difference between them — useful for understanding why two similar-looking patients received different predictions.
