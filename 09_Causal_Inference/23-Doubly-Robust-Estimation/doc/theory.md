# 23. Doubly Robust Estimation

## Learning Objectives

- Write the AIPW (augmented IPW) estimator and explain each term
- Explain "double robustness": consistent if **either** the outcome model or the propensity model is correct
- Show numerically what happens when one, both, or neither model is misspecified
- Recognize that double robustness is not a guarantee when both models are wrong

---

## 1. The Problem

G-computation (Module 19) needs the outcome model right. IPW (Module 20) needs the propensity model right. **AIPW** combines them so that you only need one.

## 2. The Concept

### 2.1 The estimator

With outcome models $\hat m_t(x)\approx E[Y\mid T=t,X=x]$ and propensity $\hat e(x)$:
$$\hat\mu_1=\frac1n\sum_i\Big[\hat m_1(X_i)+\frac{T_i\,(Y_i-\hat m_1(X_i))}{\hat e(X_i)}\Big],\quad
\hat\mu_0=\frac1n\sum_i\Big[\hat m_0(X_i)+\frac{(1-T_i)(Y_i-\hat m_0(X_i))}{1-\hat e(X_i)}\Big],$$
and $\widehat{ATE}=\hat\mu_1-\hat\mu_0$. The first term is g-computation; the second re-weights the outcome model's **residuals** to correct its error.

### 2.2 Why either model suffices

Write $\hat m_1=m_1+\delta_m$ and $\hat e=e+\delta_e$ (errors). The bias of $\hat\mu_1$ is approximately
$$E\!\left[\Big(\frac{e(X)-\hat e(X)}{\hat e(X)}\Big)\big(\hat m_1(X)-m_1(X)\big)\right],$$
a **product** of the two errors. If either error is zero, the bias vanishes. (The same product structure is why cross-fitted machine-learning nuisance models can each converge slowly, at $n^{-1/4}$, and still give root-$n$ estimates: Module 35.)

### 2.3 Caveats

If both models are wrong, AIPW can be worse than either alone, and near-zero $\hat e$ still makes it unstable. Double robustness is protection against *one* mistake, not against all of them.

## 3. Use It: Code

`code/aipw_demo.py` builds data where the true propensity score and outcome both depend on $X_1^2$, then estimates the ATE with (outcome right/wrong) × (propensity right/wrong) for g-computation, IPW and AIPW. True ATE = 2.

## Exercises

1. Which cells of the 2×2 table does AIPW fix that neither g-computation nor IPW fixes alone?
2. Make both models wrong in different ways. Is AIPW still better than the worst of the two?
3. Derive the product-of-errors bias formula for $\hat\mu_1$.

## Key Terms

| Term | What it actually means |
|---|---|
| AIPW | Outcome-model prediction plus an IPW-weighted residual correction |
| Double robustness | Consistency if at least one nuisance model is correctly specified |
| Nuisance model | A model (outcome or propensity) needed to estimate the target effect but not itself of interest |
