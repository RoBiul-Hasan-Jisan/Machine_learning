# 25. CATE and Heterogeneous Treatment Effects

## Learning Objectives

- Define the CATE $\tau(x)=E[Y(1)-Y(0)\mid X=x]$ and explain why average effects can hide important variation
- Implement the S-learner and T-learner meta-learners
- Evaluate CATE estimates when the truth is known (simulation), and explain why this is hard with real data
- State the main pitfalls: regularization bias, confounding, and false discoveries in subgroup searches

---

## 1. The Problem

An ATE of 0 may hide a treatment that helps half the population and harms the other half. **CATE** estimation asks *who* benefits, which is the basis of personalized decisions.

## 2. The Concept

### 2.1 Target

$$\tau(x)=E[Y(1)-Y(0)\mid X=x]=\mu_1(x)-\mu_0(x),\qquad \mu_t(x)=E[Y\mid T=t,X=x]\ \text{(under unconfoundedness)}.$$

### 2.2 Meta-learners

- **S-learner:** fit one model $\hat\mu(x,t)$ with $T$ as a feature; $\hat\tau(x)=\hat\mu(x,1)-\hat\mu(x,0)$. Simple, but a regularized model may ignore $T$ and shrink $\hat\tau$ toward 0.
- **T-learner:** fit $\hat\mu_1$ on treated and $\hat\mu_0$ on controls separately; $\hat\tau=\hat\mu_1-\hat\mu_0$. Captures heterogeneity but each model sees only part of the data, which adds variance.
- **X-, R-, DR-learners** improve on these (Modules 23, 35).

### 2.3 Evaluation is hard

$\tau_i$ is never observed (fundamental problem, Module 01). Only in simulation can you compute $\text{RMSE}(\hat\tau,\tau)$. With real data you rely on proxies: calibration by predicted-effect group, uplift/Qini curves, and RCT holdouts.

### 2.4 Pitfalls

- **Confounding:** heterogeneity estimators inherit the identification assumptions of Modules 06 and 14.
- **Multiple comparisons:** searching many subgroups finds spurious "heterogeneity" (Module 34 for robustness ideas).
- **Overlap** can fail in parts of covariate space, making $\hat\tau(x)$ pure extrapolation there.

## 3. Use It: Code

`code/cate_demo.py` uses a randomized experiment with $\tau(x)=1+2x_0$ and compares the S- and T-learners built on gradient boosting, reporting RMSE, correlation with the true effect, and group-wise average effects.

## Exercises

1. Add heavy noise to $Y$. Which learner degrades more?
2. Make $\tau(x)$ zero for most units and large for a few. Do both learners find it?
3. Why does regularization hurt the S-learner more than the T-learner here?

## Key Terms

| Term | What it actually means |
|---|---|
| CATE | Expected treatment effect for units with covariates $x$ |
| S-learner | One outcome model with treatment as a feature |
| T-learner | Separate outcome models per arm |
| Uplift | Industry name for an individual-level treatment effect estimate |
