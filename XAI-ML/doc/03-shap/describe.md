# 03 · SHAP (SHapley Additive exPlanations)

SHAP is the closest thing XAI has to a mathematically principled, industry-standard explanation method. It roots feature attribution in cooperative **game theory** (Shapley values, Lloyd Shapley — Nobel Prize 2012).

## What you'll learn
- The Shapley value formula and the intuition behind it (fairly splitting "credit" for a prediction among features)
- The four axioms SHAP guarantees: **Efficiency, Symmetry, Dummy, Additivity**
- Why exact computation is exponential (`2^M` subsets) and how `TreeSHAP` makes it tractable in polynomial time for tree models
- **Global explanations** (`summary_plot`) vs **local explanations** (`force_plot`) on the same model
- **Dependence plots** — how a single feature's value relates to its SHAP contribution, including interaction effects

## Read first
the Theory section below — the full mathematical treatment (formula, axioms, complexity, TreeSHAP/DeepSHAP).

## Then run
[`03-shap.ipynb`](../../code/03-shap.ipynb) — trains a Random Forest (our black box) and explains it with `shap.TreeExplainer`.

## Key takeaway
SHAP values for one prediction always sum to `prediction − average_prediction` (Efficiency axiom) — this is what makes them "additive" and auditable, unlike many other attribution methods.

## Next
[`04-lime`](../04-lime) — a faster, less theoretically strict alternative to SHAP.


---

# 03 · SHAP — Theory

### 3.1 Shapley Values (SHAP Core Theory)

**Origin:** From cooperative game theory (Lloyd Shapley, Nobel Prize 2012)
**The Problem:** Distribute a payout fairly among players based on their individual contributions.

**Mathematical Definition:**
For a prediction function $f(x)$ with features $F = \{1, 2, \dots, M\}$, the Shapley value for feature $i$ is calculated as:

$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \times [f_x(S \cup \{i\}) - f_x(S)]$$

Where:
* $S$ = subset of features without feature $i$
* $f_x(S)$ = prediction using only features in $S$ (marginalized over others)

**Four Axioms (Requirements):**
1.  **Efficiency:** The sum of all Shapley values equals the difference between the prediction and the average.
    $$\sum \phi_i = f(x) - E[f(X)]$$
2.  **Symmetry:** If two features contribute equally, they get equal Shapley values.
    If $f(S \cup \{i\}) = f(S \cup \{j\})$ for all $S$, then $\phi_i = \phi_j$
3.  **Dummy:** If a feature never changes the prediction, its Shapley value is 0.
    If $f(S \cup \{i\}) = f(S)$ for all $S$, then $\phi_i = 0$
4.  **Additivity:** For combined models, Shapley values add together.
    $$\phi_i(f+g) = \phi_i(f) + \phi_i(g)$$

**Computational Challenge:**
* Requires evaluating $2^M$ subsets (exponential time complexity).
* $M=10 \rightarrow 1024$ subsets.
* $M=100 \rightarrow$ impossible ($10^{30}$ subsets).

**Solutions:**
* **SHAP:** Approximates the values using sampling.
* **TreeSHAP:** Optimized for tree-based models (runs in polynomial time).
* **DeepSHAP:** Optimized specifically for neural networks.

---
