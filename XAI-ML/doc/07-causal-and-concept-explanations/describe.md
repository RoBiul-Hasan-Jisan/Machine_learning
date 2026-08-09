# 07 · Causal & Concept-Based Explanations — Theory

### 3.3 Information Theory Approaches

**Mutual Information:**
Measures how much knowing $X$ reduces the uncertainty about $Y$.
$$I(X;Y) = H(X) - H(X|Y) = \sum \sum P(x,y) \log\left(\frac{P(x,y)}{P(x)P(y)}\right)$$

**Feature Relevance:**
Determines the unique information a feature brings to the target prediction. Higher values equal more unique information.
$$\text{Relevance}(feature_i) = I(feature_i; target) - I(feature_i; target | \text{other features})$$

**Conditional Entropy for Explanations:**
Measures the remaining uncertainty after observing a prediction. Lower entropy means more trustworthy predictions.
$$H(\text{Target} | \text{Prediction}) = -\sum P(pred, target) \log P(target | pred)$$

---


---

## PART 5: ADVANCED TOPICS

### 5.1 Causal Explanations vs. Associative Explanations

**Association (SHAP, LIME):**
* "Feature A is correlated with prediction B."
* Does **not** imply causation.
* *Example:* Ice cream sales $\uparrow$, drowning incidents $\uparrow$ (Correlated due to summer heat, but not causal).
* **Mathematical notation:** $P(Y|X)$ (Model sees correlation).

**Causation (Causal XAI):**
* "Changing feature A will mathematically *cause* prediction B to change."
* Requires a predefined causal graph.
* Utilizes Do-calculus: $P(Y|do(X=x))$ vs. $P(Y|X=x)$.
* *Example:* "Opening an umbrella causes dryness."
* **Mathematical notation:** $P(Y|do(X))$ (Intervention effect).

**Structural Causal Models (SCM):**
Causal explanations track changes in $X$ that propagate through a structural graph.
$$X = f_X(\epsilon_X)$$
$$Y = f_Y(X, \epsilon_Y)$$

---

### 5.2 Concept-Based Explanations

**The Problem:** Standard features are often low-level and meaningless to humans (e.g., raw pixels, embeddings, numbers).
**The Solution:** Map these low-level features to high-level human concepts.

**Concept Activation Vectors (CAV):**
For a specific concept $C$ (e.g., "striped pattern"):
1.  Find data examples with and without the concept.
2.  Train a linear classifier in the model's latent embedding space.
3.  The CAV is the normal vector of the resulting decision boundary.
* *Explanation output:* Shows the sensitivity of the prediction to concept $C$.

**Testing Concept Activation Vectors (TCAV):**
Measures how heavily the model relies on the concept. A higher score means the model heavily uses that concept for its predictions.
$$\text{TCAV\_score} = \frac{|\{x: \text{sensitivity to concept} > 0\}|}{|\text{all } x|}$$
* *Example:* Concept = "Medical mask" in an X-ray. If TCAV is highly positive, the model is inappropriately using the presence of a physical mask to diagnose pneumonia.

---

### 5.3 Interactive Explanations

**Human-in-the-loop XAI:**
Creates a feedback loop: `User` $\leftrightarrow$ `Explanation Interface` $\leftrightarrow$ `Model`.
**Core Features:**
1.  **Explanatory Debugging:** The user actively challenges the provided explanation.
2.  **What-if Simulation:** The user manually modifies features to observe prediction shifts.
3.  **Contrastive Explanations:** The user asks targeted questions like "Why outcome A and not outcome B?"
4.  **Explanatory Refinement:** The user corrects factually wrong explanations to guide the model.

**Active Learning for Explanations:**
1.  Model makes a prediction and generates an explanation.
2.  User provides feedback (marks it correct/incorrect).
3.  Model updates its internal weights/logic based on this specific feedback.
4.  The next prediction and explanation inherently improve.

---

### 5.4 Uncertainty Quantification in Explanations

**Types of Uncertainty:**
* **Aleatoric Uncertainty (Data noise):** "Even with a mathematically perfect model, there is inherent randomness." (e.g., A medical test that inherently has a 2% false-positive rate).
* **Epistemic Uncertainty (Model uncertainty):** "The model is unsure because it lacks data." (e.g., The model has never seen training examples resembling this specific patient).

**Bayesian XAI:**
Instead of providing a single point-estimate explanation, it provides a probability distribution.
$$P(\text{Explanation} | \text{Data, Model})$$
* **Mean explanation:** The standard SHAP value.
* **Variance:** Represents the confidence in that explanation.

**Dropout for Uncertainty (Neural Nets):**
An engineering trick to measure epistemic uncertainty.
* Run the model 100 times on the exact same input, but with different dropout masks applied.
* Measure the variance of the SHAP values across all 100 runs.
* **Result:** High variance = the explanation is highly unreliable and the model is guessing.

