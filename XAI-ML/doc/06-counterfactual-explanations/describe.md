# 06 · Counterfactual Explanations

A counterfactual answers the most actionable question a user can ask: **"What would need to change for a different outcome?"** ("If your income were $5,000 higher, your loan would be approved.")

## What you'll learn
- The formal optimization problem: minimize `d(x, x')` subject to `f(x') = y_desired`
- Closest (L2) vs sparse (L1) vs plausible (distribution-constrained, KL-divergence) counterfactuals
- The four core properties every good counterfactual should have: **Actionability, Diversity, Proximity, Sparsity**
- A simple from-scratch counterfactual generator using random perturbation search

## Read first
the Theory section below

## Then run
[`06-counterfactual-explanations.ipynb`](../../code/06-counterfactual-explanations.ipynb) — finds the minimal feature changes that would flip a "Malignant" prediction to "Benign."

## Key takeaway
Counterfactuals are the most user-facing XAI method — they don't just say *why*, they say *what to do about it*. But naive counterfactual search (as in this notebook) can suggest implausible or non-actionable changes (e.g. "decrease age by 10 years") — production systems add plausibility and actionability constraints.

## Next
[`07-causal-and-concept-explanations`](../07-causal-and-concept-explanations) — theory on the difference between correlation-based explanations (everything so far) and true causal explanations.


---

# 06 · Counterfactual Explanations — Theory

### 3.4 Counterfactual Theory

**Definition:** The minimal change required to an input vector to successfully alter the model's prediction.

**Formal Definition:**
Find $x'$ that minimizes:
1.  Prediction $f(x') =$ desired outcome
2.  Distance $d(x, x')$ (typically L1 or L2 norms)
3.  Plausibility constraints ($x'$ must realistically exist in the data distribution)

**Optimization Problem:**
$$x' = \arg\min_{x'} d(x, x')$$
$$\text{subject to: } f(x') = y_{desired} \quad \text{and} \quad x' \in \text{feasible\_space}$$

**Types of Counterfactuals:**

* **Closest Counterfactual:** Minimizes squared Euclidean distance.
    $$x' = \arg\min_{x'} ||x - x'||_2^2 \quad \text{s.t.} \quad f(x') = y_{target}$$

* **Sparser Counterfactual (L1 regularization):** Modifies the fewest possible features.
    $$x' = \arg\min_{x'} ||x - x'||_1 \quad \text{s.t.} \quad f(x') = y_{target}$$

* **Plausible Counterfactual (Distribution constraint):** Ensures the fake data point aligns with real-world distributions using KL Divergence.
    $$x' = \arg\min_{x'} ||x - x'||_2^2 + \lambda \cdot D_{KL}(P(x') || P(data)) \quad \text{s.t.} \quad f(x') = y_{target}$$

**Core Properties:**
* **Actionability:** Modifies features that a user can realistically change (e.g., salary/debt, not age/race).
* **Diversity:** Provides multiple different possible paths (counterfactuals) to achieve the goal.
* **Proximity:** Requires the absolute minimal amount of change to the input.
* **Sparsity:** Alters the fewest number of distinct features possible.

