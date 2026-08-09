# Exercises — 08 Evaluation Metrics

1. **Faithfulness.** Using the `Completeness` formula, compute it by hand for a toy example: prediction = 0.8, sum of feature contributions = 0.75. What does the result tell you about how "complete" this explanation is?
2. **Monotonicity.** Give an example feature from the Breast Cancer dataset (e.g. `worst area`) where you would expect the monotonicity test to hold (increasing the feature should consistently increase or decrease predicted malignancy risk in one direction).
3. **Comprehensibility trade-off.** A doctor asks for an explanation with only 2 features shown (high simplicity) vs. a researcher who wants all 30 SHAP values (low simplicity, high completeness). Explain why you can't maximize both simplicity and completeness simultaneously, referencing the `Simplicity` formula.
4. **Fairness metrics.** Explain, in your own words, the difference between Demographic Parity and Equal Opportunity. Give a scenario where a model satisfies one but not the other.
5. **Robustness.** Using the `Robustness(x)` formula, explain why a model whose SHAP explanations change wildly for a tiny input perturbation (`δ`) would be considered untrustworthy, even if its raw predictions barely change.
