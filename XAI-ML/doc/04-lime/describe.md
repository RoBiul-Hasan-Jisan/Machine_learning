# 04 · LIME (Local Interpretable Model-agnostic Explanations)

LIME explains a single prediction by approximating the black box **locally** with a simple, interpretable surrogate model (usually linear).

## What you'll learn
- The LIME objective: minimize `L(f, g, π_x) + Ω(g)` — fidelity to the black box near `x`, penalized for surrogate complexity
- How the sampling process works: perturb → weight by distance → fit a local linear model
- How the kernel width `σ` trades off "local" vs "global" fidelity
- LIME vs SHAP: speed and model-agnosticism vs theoretical guarantees and stability

## Read first
the Theory section below

## Then run
[`04-lime.ipynb`](../../code/04-lime.ipynb) — explains an individual patient's prediction from the Random Forest using `lime.lime_tabular.LimeTabularExplainer`.

## Key takeaway
LIME explanations are **unstable** — rerun it on the same instance and the explanation can shift, because it's built from random perturbations. This is the central weakness the SHAP axioms were designed to avoid.

## Next
[`05-model-agnostic-pdp-ice`](../05-model-agnostic-pdp-ice) — zoom out from single predictions to *global* model behavior.


---

# 04 · LIME — Theory

### 3.2 LIME Theory (Local Interpretable Model-agnostic Explanations)

**Core Idea:** Approximate a complex "black box" model locally using a simple, highly interpretable surrogate model.

**Mathematical Formulation:**
For a specific instance $x$, find the explanation $\xi(x)$ that minimizes the objective function:

$$\xi(x) = \arg\min_{g \in G} L(f, g, \pi_x) + \Omega(g)$$

Where:
* $f$ = complex black box model
* $g$ = interpretable surrogate model (e.g., linear model)
* $\pi_x$ = locality neighborhood around $x$ (distance weighting)
* $L$ = loss function (how well $g$ approximates $f$ locally)
* $\Omega(g)$ = complexity penalty (keeps $g$ simple)

**Sampling Process:**
1.  Start with the instance $x$.
2.  Sample new instances $z$ around $x$ (using Gaussian noise).
3.  Weight these samples by their distance from $x$: $\pi_x(z) = \exp\left(-\frac{D(x,z)^2}{\sigma^2}\right)$
4.  Get predictions $f(z)$ from the black box model.
5.  Train the linear model $g$ on these perturbed samples.
6.  The coefficients of $g$ become the feature importance scores.

**Locality Weighting:**
* **Distance:** $||x - z||$ (Euclidean distance)
* **Weight:** $w(x,z) = \exp\left(-\frac{||x - z||^2}{\sigma^2}\right)$
* $\sigma$ controls the locality width:
    * **Small $\sigma$:** Very local (spiky focus).
    * **Large $\sigma$:** More global (smooth focus).

**Advantages over SHAP:**
* Significantly faster for large datasets.
* Completely model-agnostic (works with any model).
* Can utilize any interpretable surrogate model (linear, tree, rules).

**Disadvantages:**
* Unstable (different random samples $\rightarrow$ different explanations).
* Highly sensitive to the $\sigma$ (kernel width) parameter.
* Lacks the strict theoretical guarantees of SHAP.

---
