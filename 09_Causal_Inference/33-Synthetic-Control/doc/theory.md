# 33. Synthetic Control

## Learning Objectives

- Explain how a synthetic control builds a counterfactual for **one** treated unit from a weighted combination of untreated donors
- Fit the weights by constrained least squares on pre-treatment data and read the post-treatment gap as the effect
- Understand the factor-model assumption behind it and why a good pre-treatment fit matters
- Run placebo-in-space inference

---

## 1. The Problem

A single state passes a law, a single country adopts a policy. There is one treated unit, so DiD with one control is fragile: which control is "parallel"? **Synthetic control** (Abadie and Gardeazabal 2003; Abadie, Diamond, Hainmueller 2010) constructs the comparison from many donors in a data-driven way.

## 2. The Concept

### 2.1 Estimator

Unit 1 is treated from period $T_0+1$; units $2,\dots,J+1$ are donors. Choose weights $w_j\ge0$, $\sum_j w_j=1$ to minimize the pre-treatment discrepancy
$$\sum_{t\le T_0}\Big(Y_{1t}-\sum_{j}w_jY_{jt}\Big)^2.$$
The estimated effect at $t>T_0$ is the gap $\hat\tau_t=Y_{1t}-\sum_jw_j^*Y_{jt}$.

### 2.2 Why nonnegative weights summing to one

The constraints keep the synthetic control an interpolation of real donors (no extrapolation), make the weights readable (which units resemble the treated unit) and regularize the fit.

### 2.3 Identification

Assume outcomes follow a factor model $Y_{jt}(0)=\delta_t+\lambda_t^\top\mu_j+\varepsilon_{jt}$. If some convex combination of donors matches the treated unit's pre-treatment path *and* the factor loadings $\mu_1$, it will also track the treated unit's untreated path afterward. A poor pre-treatment fit means the method is not credible for that unit; do not extrapolate.

### 2.4 Inference: placebo in space

Apply the same procedure to each donor as if it were treated (using the other donors). Compare the treated unit's post/pre RMSPE ratio with the placebo distribution; the rank gives a permutation-style $p$-value, roughly $1/(J+1)$ at best.

## 3. Use It: Code

`code/synthetic_control_demo.py` generates 30 donors from a 3-factor model, builds a treated unit as a mix of four donors plus an effect of 3 after $T_0=25$, fits the weights with SLSQP, reports pre-fit quality, the estimated gap versus the truth (expect it to be close but not exact: the weights are not uniquely identified with 30 donors and 3 factors), and the placebo-in-space test.

## Exercises

1. Make the treated unit lie outside the donors' convex hull. What happens to the pre-fit and the estimate?
2. Reduce the number of pre-treatment periods to 5. Effect on the fit?
3. Which weights make the synthetic control equal the simple control-group average (the comparison group DiD would use)? What does that say about how synthetic control generalizes it?

## Key Terms

| Term | What it actually means |
|---|---|
| Donor pool | Untreated units eligible to form the synthetic control |
| Pre-treatment fit (RMSPE) | How closely the synthetic path tracks the treated unit before treatment |
| Placebo in space | Re-running the method on untreated units to build a reference distribution |
