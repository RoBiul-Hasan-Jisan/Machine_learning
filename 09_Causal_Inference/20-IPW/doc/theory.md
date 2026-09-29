# 20. Inverse Probability Weighting (IPW)

## Learning Objectives

- Derive IPW from the identity $E[Y(1)] = E[TY/e(X)]$
- Explain how weighting creates a pseudo-population in which $T\perp X$
- Compute Horvitz-Thompson and Hajek (stabilized) estimators
- Diagnose extreme weights with the effective sample size and use truncation

---

## 1. The Problem

G-computation models the outcome. IPW models the **treatment mechanism** instead: the propensity score $e(x)=P(T=1\mid X=x)$.

## 2. The Concept

### 2.1 The identity

Under exchangeability, consistency and positivity,
$$E\!\left[\frac{T\,Y}{e(X)}\right]=E\!\left[\frac{T\,Y(1)}{e(X)}\right]=E\!\left[E\!\left[\frac{T}{e(X)}\Big|X,Y(1)\right]Y(1)\right]=E[Y(1)],$$
because $E[T\mid X,Y(1)]=e(X)$. Similarly $E[(1-T)Y/(1-e(X))]=E[Y(0)]$.

### 2.2 The pseudo-population

Weight each unit by $w_i=1/P(T_i\mid X_i)$. Treated units with a small chance of treatment count a lot; the weighted sample behaves as if treatment were randomized: $T\perp X$.

### 2.3 Estimators

Horvitz-Thompson: $\hat\mu_1=\frac1n\sum \frac{T_iY_i}{\hat e_i}$. Hajek (self-normalized): $\hat\mu_1=\frac{\sum T_iY_i/\hat e_i}{\sum T_i/\hat e_i}$, usually more stable.

### 2.4 Weight diagnostics

The effective sample size $ESS=(\sum w_i)^2/\sum w_i^2$ shrinks when weights are extreme (near-positivity violation). **Truncating** (clipping $\hat e$ to $[c,1-c]$) trades bias for variance. The trade is not always small: in §3's strong-confounding setting, clipping at $c=0.05$ cuts the standard deviation about five-fold but introduces a large bias, because many units genuinely have propensity scores below 0.05. Choose $c$ deliberately and report results with and without it.

## 3. Use It: Code

`code/ipw_demo.py` estimates propensity scores by logistic regression, compares Horvitz-Thompson and Hajek, and shows how a strong $X\to T$ relationship produces extreme weights, low ESS, and the bias–variance effect of truncation over repeated samples.

## Exercises

1. Verify that weighted covariate means are balanced between arms.
2. Vary the strength of $X\to T$ and plot ESS.
3. Explain why a propensity score of exactly 0 or 1 makes IPW undefined.

## Key Terms

| Term | What it actually means |
|---|---|
| Propensity score | $e(x)=P(T=1\mid X=x)$ |
| Pseudo-population | Weighted sample in which treatment is independent of covariates |
| Effective sample size | $(\sum w)^2/\sum w^2$; information left after weighting |
