# 29. Time-varying Confounding

## Learning Objectives

- Define treatment–confounder feedback and explain why ordinary regression adjustment fails there
- State sequential exchangeability and positivity for a sequence of treatments
- Implement the parametric g-formula and stabilized-weight IPW for a two-period treatment
- Verify both against the truth computed by simulating the intervention

---

## 1. The Problem

Treatments often repeat over time: a drug dose each month, a policy each year. A covariate $L_1$ measured after the first treatment $A_0$ can both **confound** the next treatment $A_1$ (doctors treat sicker patients) *and* be **affected by** $A_0$. This is *treatment–confounder feedback*.

## 2. The Concept

### 2.1 Why standard adjustment breaks

DAG: $A_0\to L_1\to A_1$, $L_1\to Y$, $A_0\to Y$, $A_1\to Y$, and an unmeasured $U\to L_1$, $U\to Y$.

- **Not adjusting** for $L_1$ leaves the backdoor path $A_1\leftarrow L_1\to Y$ open.
- **Adjusting** for $L_1$ (a) blocks the part of $A_0$'s effect that goes through $L_1$, and (b) conditions on a collider ($A_0\to L_1\leftarrow U$), opening $A_0\leftarrow\!\!\cdot\ \cdot\to U\to Y$.

Neither choice identifies the effect of the treatment *strategy*. A single set of covariates cannot work; you need **g-methods**.

### 2.2 Assumptions for a treatment sequence $\bar a=(a_0,a_1)$

- **Sequential exchangeability:** $Y(\bar a)\perp A_k\mid \bar A_{k-1},\bar L_k$ for each $k$.
- **Positivity:** every treatment history has positive probability at every covariate history.
- **Consistency:** the observed outcome equals $Y(\bar a)$ when $\bar A=\bar a$.

### 2.3 Parametric g-formula

$$E[Y(a_0,a_1)]=\sum_{l_1}E[Y\mid A_0{=}a_0,L_1{=}l_1,A_1{=}a_1]\;P(L_1{=}l_1\mid A_0{=}a_0).$$
Fit the outcome model and the covariate model, then plug in the strategy. It handles feedback because $L_1$ is drawn from its distribution *under $a_0$* rather than held fixed.

### 2.4 Inverse probability weighting (marginal structural models)

Weight each person by
$$w_i=\frac{1}{P(A_0{=}a_{0i})\,P(A_1{=}a_{1i}\mid A_{0i},L_{1i})}.$$
In the weighted pseudo-population, $L_1$ no longer predicts $A_1$, but $A_0$'s effect through $L_1$ is preserved. Unstabilized weights can be huge; **stabilized** weights put $P(A_1\mid A_0)$ in the numerator.

## 3. Use It: Code

`code/time_varying_demo.py` simulates the DAG above. Truth is computed by intervening on $(A_0,A_1)$. It compares the naive contrast, regression adjusting for $L_1$, the g-formula, and IPW for "always treat" vs "never treat".

## Exercises

1. Compute the effect of $(A_0{=}1,A_1{=}0)$ vs $(A_0{=}0,A_1{=}0)$ with the g-formula.
2. Truncate the IPW weights. What happens to bias and variance?
3. Why does $L_1$ need to be measured *after* $A_0$ and *before* $A_1$?

## Key Terms

| Term | What it actually means |
|---|---|
| Treatment–confounder feedback | A time-varying confounder is itself affected by past treatment |
| G-methods | G-formula, IPW for marginal structural models, g-estimation |
| Sequential exchangeability | No unmeasured confounding at each time given the observed history |
