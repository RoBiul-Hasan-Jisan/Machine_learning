#  Causal Machine Learning: Double/Debiased ML

## Learning Objectives

- Explain why plugging a flexible ML model into a causal estimator causes regularization and overfitting bias
- Derive the partialling-out (Frisch–Waugh–Lovell) form of the partially linear model
- Implement double/debiased machine learning (DML) with cross-fitting
- Know what Neyman orthogonality buys you, and its limits

---

## 1. The Problem

Modules 19–23 needed models for $E[Y\mid T,X]$ or $P(T\mid X)$. With many covariates we want random forests or boosting, but ML estimators are **biased** (regularization) and **overfit**. Plugging them in naively contaminates the causal estimate.

## 2. The Concept

### 2.1 Partially linear model

$$Y=\theta T+g(X)+\varepsilon,\qquad T=m(X)+v,\qquad E[\varepsilon\mid T,X]=0,\ E[v\mid X]=0,$$
where $\theta$ is the causal effect (under unconfoundedness given $X$) and $g,m$ are unknown nuisance functions.

### 2.2 Partialling out

Let $\ell(X)=E[Y\mid X]=\theta m(X)+g(X)$. Then
$$Y-\ell(X)=\theta\,(T-m(X))+\varepsilon,$$
so $\theta$ is the coefficient from regressing the **outcome residual** on the **treatment residual**:
$$\hat\theta=\frac{\sum_i\tilde v_i\,\tilde y_i}{\sum_i\tilde v_i^2},\quad \tilde y=Y-\hat\ell(X),\ \tilde v=T-\hat m(X).$$

### 2.3 Neyman orthogonality

The score $\psi(\theta;\ell,m)=(Y-\ell(X)-\theta(T-m(X)))(T-m(X))$ has zero derivative with respect to the nuisance functions at the truth, so first-order errors in $\hat\ell,\hat m$ do not bias $\hat\theta$; only the *product* of their errors matters (compare Module 23). Hence nuisance rates of $n^{-1/4}$ suffice for a root-$n$ estimate of $\theta$.

### 2.4 Cross-fitting

Overfitting makes residuals $\tilde v$ correlate with noise. **Cross-fitting** fixes this: split the data into $K$ folds, fit $\hat\ell,\hat m$ on all folds but one, compute residuals on the held-out fold, then pool the residuals. Each observation's residual comes from a model that never saw it.

### 2.5 Limits

DML still requires unconfoundedness and overlap; ML cannot fix an unmeasured confounder. It also targets an average parameter; heterogeneity needs further methods (Module 25).

## 3. Use It: Code

`code/dml_demo.py` uses 10 covariates with nonlinear confounding and true $\theta=1$. Over repeated samples it compares: naive regression, linear adjustment, a random-forest plug-in, DML without cross-fitting and DML with 5-fold cross-fitting. (The demo takes about a minute. Here the forests are regularized, so DML without cross-fitting is already close to the cross-fitted version; the gap widens with more flexible learners such as deep or unpruned trees, which is Exercise 4.)

## Exercises

1. Change the number of folds (2, 5, 10). How stable is $\hat\theta$?
2. Swap the random forest for gradient boosting.
3. Why does the naive plug-in random forest tend to shrink $\hat\theta$ toward zero?
4. Set `min_samples_leaf=1` (unpruned trees). Compare DML with and without cross-fitting.

## Key Terms

| Term | What it actually means |
|---|---|
| DML | Estimate the effect from residualized outcome and treatment, using ML nuisance models with cross-fitting |
| Neyman orthogonality | Insensitivity of the estimating equation to small nuisance errors |
| Cross-fitting | Predict each fold with models trained on the other folds |
