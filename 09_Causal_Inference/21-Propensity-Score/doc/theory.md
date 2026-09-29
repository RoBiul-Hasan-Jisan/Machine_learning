# 21. The Propensity Score

## Learning Objectives

- Define the propensity score and state the balancing-score theorem
- Explain why conditioning on the scalar $e(X)$ is enough when conditioning on all of $X$ is
- Check covariate balance with standardized mean differences (SMD) and check overlap
- Estimate an ATE by stratifying on the propensity score

---

## 1. The Problem

Stratifying on many covariates fails (Module 07 §3.1: curse of dimensionality). Rosenbaum and Rubin (1983) showed all of $X$ can be collapsed into **one number**.

## 2. The Concept

### 2.1 Definition and theorem

$$e(x)=P(T=1\mid X=x).$$

**Balancing property:** $T\perp X\mid e(X)$. Among units with the same propensity score, treated and control units have the same covariate distribution.

**Consequence:** if $Y(t)\perp T\mid X$ (unconfoundedness), then $Y(t)\perp T\mid e(X)$. Adjusting for the scalar $e(X)$ removes confounding by all of $X$.

*Sketch:* $P(T=1\mid X,e(X))=P(T=1\mid X)=e(X)$, which depends on $X$ only through $e(X)$, so $T\perp X\mid e(X)$. Then $E[Y(t)\mid T=t,e]=E[Y(t)\mid e]$ by averaging the conditional independence over $X$ given $e$.

### 2.2 The propensity score is a *design* tool

You estimate $e(X)$ (usually logistic regression) **without looking at the outcome**, check balance, iterate on the model, and only then estimate effects. The goal of the model is balance, not prediction accuracy: a highly predictive $\hat e$ can even hurt overlap.

### 2.3 Diagnostics

- **Standardized mean difference:** $SMD=\dfrac{\bar x_1-\bar x_0}{\sqrt{(s_1^2+s_0^2)/2}}$; a common rule of thumb is $|SMD|<0.1$ after adjustment.
- **Overlap:** compare the distribution of $\hat e$ in treated and control; regions with only one group violate positivity.

### 2.4 Stratification on the score

Cut $\hat e$ into $K$ strata (quintiles: Cochran showed five removes most bias), take the treated–control outcome difference in each, and average weighted by stratum size, exactly Module 07 with the score as the stratifying variable.

## 3. Use It: Code

`code/propensity_score_demo.py` estimates $\hat e$ from three covariates, reports SMDs raw, after IPW, and after quintile stratification, and estimates the ATE by score stratification (truth = 2).

## Exercises

1. Add a covariate that affects $T$ but is measured with noise. What happens to balance?
2. Increase the number of strata to 20. What happens to bias and variance?
3. Why can a perfectly predictive propensity model be a warning sign?

## Key Terms

| Term | What it actually means |
|---|---|
| Balancing score | A function $b(X)$ with $T\perp X\mid b(X)$; the propensity score is the coarsest one |
| SMD | Scale-free difference in covariate means between arms |
| Overlap | Both arms are observed across the range of $\hat e$ |
