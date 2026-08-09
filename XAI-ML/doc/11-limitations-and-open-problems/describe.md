# 11 · Limitations, Open Problems & Practical Guidelines

## PART 8: LIMITATIONS AND OPEN PROBLEMS

### 8.1 Current Limitations
* **Explanation ≠ Justification:** An explanation might state, "The model used feature X," but it cannot answer if it was ethically or logically *right* to use that feature.
* **Stability Problem:** A slight change in the underlying data can lead to completely different SHAP values. Which explanation is the "true" one?
* **Human Over-trust:** Studies show users trust models *more* when given an explanation, even when the explanation clearly reveals the model's logic is flawed or wrong.
* **Manipulation Risk:** Adversarial examples can be crafted to fool both the model and the explanation system simultaneously (i.e., the explanation can be hacked).

### 8.2 Open Research Questions
1.  **Causal XAI:** How do we transition from purely associative explanations to true causal explanations?
2.  **Interactive Explanations:** How can we better incorporate human feedback into the explanation loop to correct models?
3.  **Multi-objective Explanations:** How do we mathematically balance competing goals like accuracy, simplicity, stability, and fairness?
4.  **Explanation Verification:** How can we definitively prove an explanation is factually correct?
5.  **Scalable XAI:** How do we compute explanations for trillion-parameter foundation models efficiently?

---


---

## PART 9: PRACTICAL GUIDELINES

### 9.1 When to Use Which Method

| Scenario | Best Method | Why |
| :--- | :--- | :--- |
| **Regulatory Compliance** | SHAP | Provides strict mathematical guarantees. |
| **Real-time Predictions** | LIME | Highly optimized for speed. |
| **Debugging Model** | Global Importance | Gives a broad overview of model behavior. |
| **User-facing Explanation** | Counterfactuals | Provides highly actionable next steps. |
| **Medical Diagnosis** | Multiple Methods | Allows for cross-validation of explanations. |
| **Research Publication** | SHAP + LIME | Serves as standard benchmark baselines. |

### 9.2 Explanation Checklist

**Before Deploying XAI:**
* Define the audience (Domain expert vs. general public).
* Define the use case (Debugging the model vs. building user trust).
* Set specific success metrics (Fidelity, stability, comprehensibility).
* Choose appropriate methods (Use at least 2 to cross-validate).
* Validate the generated explanations with actual domain experts.

**Quality Assurance Example:**
```python
# Check explanation quality
def validate_explanation(model, x, explanation):
    checks = {
        "fidelity": abs(model.predict(x) - sum(explanation)) < 0.1,
        "stability": similar_outputs_for_similar_inputs(x),
        "simplicity": len(explanation.features) <= 5,
        "actionability": all_features_are_changeable(explanation)
    }
    return all(checks.values())

```


##  Explainable AI (XAI) Summary Table

| Concept                        | Definition                                                    | Key Methods                                                             | Common Evaluation Metrics                               |
|--------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------|
| **Feature Attribution**        | Quantifies contribution of each input feature to a prediction | SHAP, LIME, Integrated Gradients                                        | Fidelity, Consistency, Feature Importance Correlation |
| **Local Explanation**          | Explains why a specific instance got a prediction             | LIME, SHAP (local), Counterfactual explanations                         | Fidelity, Stability, Local Accuracy                   |
| **Global Explanation**         | Explains overall model behavior across dataset                | Permutation Importance, Partial Dependence Plot (PDP), Surrogate models | Comprehensibility, Global Fidelity                    |
| **Counterfactual Explanation** | Finds minimal changes to flip prediction                      | Optimization-based CF, DiCE, Wachter method                             | Proximity, Sparsity, Plausibility, Validity           |
| **Causal Explanation**         | Explains cause-effect relationships using interventions       | Structural Causal Models (SCM), Do-calculus                             | Causal Validity, Intervention Consistency             |
| **Concept-Based Explanation**   | Explains using human-understandable concepts                  | TCAV, Concept Bottleneck Models                                         | Concept Sensitivity, Interpretability Score           |
| **Fairness Explanation**       | Ensures model does not discriminate across groups             | Fairness-aware SHAP, Bias detection methods                             | Demographic Parity, Equal Opportunity, Disparate Impact |