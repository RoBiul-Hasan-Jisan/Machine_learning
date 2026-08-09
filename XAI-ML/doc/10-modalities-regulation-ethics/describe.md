# 10 · XAI Across Modalities, Regulation & Ethics

## PART 6: XAI IN DIFFERENT MODALITIES

### 6.1 Tabular Data XAI
* **Feature Attribution:** SHAP, LIME (industry standard for tabular data).
* **Feature Interactions:** 2-way SHAP (measuring how features depend on each other).
* **Decision Rules:** Anchors (high-precision local rules).

### 6.2 Image XAI
* **Saliency Maps:** Highlights which pixels matter most for the prediction.
* **Grad-CAM:** Gradient-based localization (highlights regions of an image).
* **Integrated Gradients:** Path-based attribution tracking from a baseline image.
* **LIME for Images:** Superpixel segmentation (grouping pixels into interpretable patches).

> **Example:** Dog Breed Classification
> * **Saliency shows:** Ears, fur texture, face shape.
> * **Explanation output:** "Ears contributed 60%, face shape 30% to the prediction."

### 6.3 Text XAI
* **Attention Visualization:** Highlights which specific words matter most.
* **LIME for Text:** Evaluates prediction shifts when specific words are removed/kept.
* **Integrated Gradients:** Tracks embedding gradients back to the original tokens.

> **Example:** Sentiment Analysis
> * **Input:** "This movie was absolutely INCREDIBLE!"
> * **Attention on:** "INCREDIBLE" (strong positive weight).
> * **Interaction:** "not" + "good" = negative phrasing identified.

### 6.4 Time Series XAI
* **Temporal SHAP:** Tracks feature importance dynamically over a timeline.
* **Counterfactuals:** Calculates how changing past events alters future predictions.
* **Pattern Attributions:** Identifies which specific subsequences or trends matter most.

---


---

## PART 7: REGULATORY AND ETHICAL FRAMEWORKS

### 7.1 GDPR Right to Explanation (EU)
**Article 22: Automated decision-making**
Grants users the right to not be subject to purely automated decisions, the right to human review, and the right to meaningful information about the system's logic.

**Requirements for any automated decision:**
1.  Explanation of system functionality.
2.  The weight of each deciding factor.
3.  Reasons for the specific outcome.
4.  Clear instructions on how to challenge the decision.

### 7.2 FDA Guidelines for Medical AI
**Explainability Requirements:**
* **Tier 1 (High-risk):** Full interpretability (glass-box models only).
* **Tier 2 (Medium-risk):** Explainable black box (requires robust post-hoc explanations).
* **Tier 3 (Low-risk):** Validation only (statistical proof of safety).

> **Example:** Cancer Diagnosis AI
> * Must explicitly explain which image features indicate cancer.
> * Must provide clear uncertainty estimates.
> * Must be validated by human experts.

### 7.3 Algorithmic Accountability Act (US)
**Core Requirements:**
1.  Impact assessment must be conducted before deployment.
2.  Ongoing monitoring for algorithmic bias.
3.  Direct explanations provided to affected individuals.
4.  Right to appeal automated decisions.

---
