# 34. Sensitivity Analysis

## Learning Objectives

- Explain why unconfoundedness cannot be tested and what sensitivity analysis offers instead
- Compute the E-value for a risk ratio and interpret it
- Use the omitted-variable-bias formula for linear models
- Compute and interpret the Cinelli–Hazlett robustness value

---

## 1. The Problem

Modules 06 and 17 showed that "no unmeasured confounding" is untestable. Sensitivity analysis asks a different question: **how strong would unmeasured confounding have to be to change our conclusion?**

## 2. The Concept

### 2.1 E-value (VanderWeele and Ding, 2017)

For an observed risk ratio $RR>1$, the E-value is the minimum strength of association, on the risk-ratio scale, that an unmeasured confounder would need with **both** treatment and outcome (conditional on measured covariates) to fully explain away the observed association:
$$E=RR+\sqrt{RR\,(RR-1)}.$$
For $RR<1$ use $1/RR$. Apply the same formula to the confidence limit closest to 1 to judge whether the interval could include 1. A large E-value means only a very strong confounder could explain the result; it does not say such a confounder is absent.

### 2.2 Omitted variable bias (linear model)

If the true model is $Y=\tau T+\gamma U+\varepsilon$ and we regress $Y$ on $T$ only,
$$\hat\tau_{short}\to\tau+\gamma\,\delta,\qquad \delta=\frac{\text{Cov}(T,U)}{\text{Var}(T)}.$$
Bias = (effect of $U$ on $Y$) × (association of $U$ with $T$). A sensitivity table shows $\hat\tau-\gamma\delta$ over plausible $(\gamma,\delta)$.

### 2.3 Robustness value (Cinelli and Hazlett, 2020)

With $t$-statistic $t$ and $df$ residual degrees of freedom for the treatment coefficient, let $f=|t|/\sqrt{df}$. The **robustness value** for reducing the estimate to zero is
$$RV=\tfrac12\Big(\sqrt{f^4+4f^2}-f^2\Big).$$
Interpretation: an unmeasured confounder that explains at least $RV$ of the residual variance of **both** treatment and outcome could bring the estimate to zero; a weaker one could not. Benchmark $RV$ against the partial $R^2$ of strong *observed* covariates.

### 2.4 Limits

These are bounds under stated parametrizations, not tests. They depend on judging which strengths are plausible, which requires domain knowledge.

## 3. Use It: Code

`code/sensitivity_demo.py` (i) computes E-values for an example risk ratio and CI, (ii) simulates omitted-variable bias and checks the formula, and (iii) computes the robustness value for a regression and compares it with an observed covariate's strength.

## Exercises

1. Compute the E-value for $RR=0.6$.
2. Explain why the E-value of the CI limit can be 1 even when the point estimate has a large E-value.
3. In (ii), vary $\gamma$ and $\delta$ and verify the bias grows with their product.

## Key Terms

| Term | What it actually means |
|---|---|
| E-value | Minimum confounder strength (RR scale) needed to explain away an association |
| Omitted variable bias | Bias of a regression coefficient from leaving out a confounder |
| Robustness value | Partial-$R^2$ strength of confounding needed to nullify an estimate |
