# 19. G-computation

## Learning Objectives

- State the g-computation (parametric g-formula) algorithm as four concrete steps
- Explain why it is the adjustment formula (Module 18) turned into an algorithm that handles many, continuous covariates
- See how outcome-model misspecification biases the estimate, and how flexible models help
- Attach uncertainty with the bootstrap

---

## 1. The Problem

The adjustment formula sums over strata of $X$. With many or continuous covariates that is impossible directly. **G-computation** replaces the sum with a fitted outcome model and a sample average.

## 2. The Concept

### 2.1 The algorithm

1. Fit an outcome model $\hat m(t,x)\approx E[Y\mid T=t,X=x]$ on the observed data.
2. Set $T=1$ for **everyone** and predict $\hat m(1,X_i)$; set $T=0$ for everyone and predict $\hat m(0,X_i)$.
3. Average each set of predictions: $\hat\mu_t=\frac1n\sum_i\hat m(t,X_i)$.
4. Report $\widehat{ATE}=\hat\mu_1-\hat\mu_0$.

Under consistency, exchangeability and positivity, $\hat\mu_t$ estimates $E[Y(t)]$ (Module 18 §2.1).

### 2.2 Why "set T for everyone" is the do-operator in code

Step 2 is the graph surgery of Module 03 performed on a fitted model: covariates keep their observed values while $T$ is overwritten. Subsetting instead (only treated units) would be the naive conditional.

### 2.3 Model dependence

G-computation is only as good as $\hat m$. If the truth is nonlinear and $\hat m$ is linear, the average of predictions is biased. Flexible learners reduce this risk; Module 23 shows how to get protection against outcome-model error.

### 2.4 Uncertainty

Resample rows, refit, recompute the ATE, and take percentile intervals (the bootstrap). The analytic variance ignores the estimation of $\hat m$.

## 3. Use It: Code

`code/gcomputation_demo.py` uses $Y(0)=X_1+0.5X_1^2+X_2+\varepsilon$ and $Y(1)=Y(0)+1+0.5X_1$ with treatment depending on $X$. True ATE $=1$. It compares naive, a misspecified linear g-computation, and a flexible (quadratic + interaction) g-computation with a bootstrap interval.

## Exercises

1. Replace the outcome model with gradient boosting. Does bias change?
2. Compute the ATT by averaging only over treated units in step 3.
3. Why must step 2 use the observed covariates rather than resampling them?

## Key Terms

| Term | What it actually means |
|---|---|
| G-computation | Fit outcome model, predict under each treatment for all units, average |
| Standardization | Same estimator viewed as re-weighting to the population covariate distribution |
| Bootstrap | Resampling rows to approximate the sampling distribution of an estimator |
