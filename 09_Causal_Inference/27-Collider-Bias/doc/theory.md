# 27. Collider Bias

## Learning Objectives

- Explain collider bias as conditioning on a common effect, including via restriction and via descendants
- Recognize **M-bias**: a case where adjusting for a pre-treatment covariate *creates* bias
- Reconcile M-bias with the advice "adjust for pre-treatment covariates" (Module 15)
- Quantify how the bias grows with the strength of the collider's causes

---

## 1. The Problem

Modules 08 and 11 introduced colliders. Two practical questions remain: how often do they show up in ordinary analyses, and can a **pre-treatment** variable be one?

## 2. The Concept

### 2.1 Three ways to condition on a collider

1. **Regression adjustment:** include the collider as a covariate.
2. **Restriction / selection:** analyze only units with $C=1$ (patients admitted, respondents who answered, people who became celebrities). This is Berkson's paradox.
3. **Conditioning on a descendant** of the collider, which partially opens the path (Module 11 §2.3).

### 2.2 M-bias

DAG: $T\leftarrow U_1\rightarrow M\leftarrow U_2\rightarrow Y$, with $U_1,U_2$ unmeasured and **no** effect of $T$ on $Y$ needed for the point. $M$ is measured before treatment and is associated with both $T$ and $Y$, so it looks like a textbook confounder. But the only path between $T$ and $Y$ runs through the collider $M$, so it is already blocked. **Adjusting for $M$ opens it** and introduces bias where there was none.

### 2.3 How to reconcile with "adjust for pre-treatment covariates"

The disjunctive cause criterion (Module 15) says to adjust for pre-treatment causes of $T$ or $Y$. $M$ is a cause of *neither*; it is an effect of the unmeasured causes. That is the whole difference between "pre-treatment" and "cause of treatment or outcome". Simulation studies suggest M-bias is usually small compared with the bias from omitting real confounders, but it is not zero, and it grows with the strength of $U_1\to M$ and $U_2\to M$.

### 2.4 Magnitude

In a linear-Gaussian version, the bias after adjusting for $M$ scales with $\rho_{U_1M}\,\rho_{U_2M}$: strong causes of the collider mean a strong induced association.

## 3. Use It: Code

`code/collider_bias_demo.py` simulates M-bias (true effect 1) at several collider strengths, a Berkson's-paradox example (two independent diseases and hospital admission), and conditioning on a descendant of a collider.

## Exercises

1. Add a weak direct path $U_1\to Y$. Does adjusting for $M$ now help or hurt?
2. Why is the naive (unadjusted) estimate unbiased in the M-bias DAG?
3. Give a survey or clinical example of restriction on a collider.

## Key Terms

| Term | What it actually means |
|---|---|
| Collider | A variable with two arrowheads pointing into it on a path |
| M-bias | Bias introduced by adjusting for a pre-treatment collider |
| Berkson's paradox | Spurious negative association among units selected on a common effect |
