# 08 · Evaluation Metrics for XAI

## PART 4: EVALUATION METRICS FOR XAI

### 4.1 Faithfulness (Accuracy of Explanation)

**Definition:** The explanation should truthfully and accurately represent the model's actual internal behavior.

**Metrics:**
1.  **Explanation Fidelity:** $\text{correlation}(\text{explanation\_weights}, \text{true\_gradient})$
2.  **Completeness:** $1 - \frac{|\text{prediction} - \sum \text{contributions}|}{|\text{prediction}|}$
3.  **Consistency:** Same inputs $\rightarrow$ similar explanations.
4.  **Stability:** Small input change $\rightarrow$ small explanation change.

**Monotonicity Test:**
For any feature $i$: If the feature's value increases $\rightarrow$ the prediction increases, then the explanation weight must be positive.
* **Monotonicity Score:** The percentage of features satisfying this logical rule.

---

### 4.2 Comprehensibility (Ease of Understanding)

**Cognitive Load Metrics:**
* **Simplicity:** $1 / (\text{number of features in the explanation})$
* **Rule Length:** The average length of decision rules (shorter is better).
* **Concept Complexity:** The total number of distinct concepts presented to the user.
* **Human Response Time:** Seconds required for a human to understand the explanation (lower is better).

**User Studies Criteria:**
* **Accuracy:** Can the person accurately predict the model's output based on the explanation?
* **Confidence:** How sure is the person about their understanding of the model?
* **Simulation:** Can the person manually simulate the model's reasoning process?
* **Contrast:** Can the person successfully identify specific feature influences?

---

### 4.3 Fairness Metrics

**Group Fairness:**
* **Demographic Parity:** $P(\hat{Y}=1|A=a) = P(\hat{Y}=1)$ (Equal acceptance rates across groups).
* **Equal Opportunity:** $P(\hat{Y}=1|Y=1,A=a) = P(\hat{Y}=1|Y=1)$ (Equal true positive rates across groups).
* **Equalized Odds:** $P(\hat{Y}=1|Y=y,A=a) = P(\hat{Y}=1|Y=y)$ for all $y$.

**Individual Fairness:**
* Similar individuals $\rightarrow$ Similar predictions.
* **Fairness Score:** $1 - \max_{\text{similar } x_1,x_2} |f(x_1) - f(x_2)|$

**Explanations for Fairness:**
* Disparate impact detection via SHAP values.
* Bias identification by analyzing global feature importance.
* Counterfactual fairness (e.g., asking: "Would the outcome change if the protected attribute changed?").

---

### 4.4 Robustness Metrics

**Explanation Robustness:**
Measures how much an explanation changes when the input is slightly modified. Lower scores mean the explanation is more robust and doesn't wildly change due to data noise.
$$\text{Robustness}(x) = E[||E(x) - E(x+\delta)||] \quad \text{over small perturbations } \delta$$

**Adversarial Robustness:**
Measures vulnerability to intentional manipulation. Higher scores indicate the explanation system can be tricked.
* **Attack Success:** The percentage of instances where an attacker successfully changes the prediction while keeping the explanation visually/mathematically similar.

---
