# 18. The Adjustment Formula

## Learning Objectives

- Derive the adjustment (standardization) formula from unconfoundedness and consistency
- Compute it exactly from a probability table, and contrast it with the naive conditional
- Extend it to continuous covariates and see why it turns into "regress, then average"
- State when it fails (positivity)

---

## 1. The Problem

Module 14 stated the backdoor adjustment formula. Here we derive it from potential outcomes so the two languages meet, and then show how it is actually computed.

## 2. The Concept

### 2.1 Derivation

Assume consistency ($Y=Y(t)$ when $T=t$), conditional exchangeability $Y(t)\perp T\mid X$, and positivity $0<P(T=1\mid X=x)<1$. Then

$$E[Y(t)] = \sum_x E[Y(t)\mid X=x]P(x) \overset{\text{exch.}}{=} \sum_x E[Y(t)\mid T=t,X=x]P(x) \overset{\text{cons.}}{=} \sum_x E[Y\mid T=t,X=x]\,P(x).$$

Every term on the right is observable. The naive quantity differs only in the weights: $E[Y\mid T=t] = \sum_x E[Y\mid T=t,X=x]\,\underbrace{P(x\mid T=t)}_{\ne P(x)}$. **Adjustment re-weights each stratum by its population share instead of its share within the treatment group.**

### 2.2 Exact worked example

Let $Z\in\{0,1\}$ with $P(Z=1)=0.4$; $P(T=1\mid Z=0)=0.2$, $P(T=1\mid Z=1)=0.7$; and $P(Y=1\mid T,Z)$:

| | $Z=0$ | $Z=1$ |
|---|---|---|
| $T=0$ | 0.10 | 0.40 |
| $T=1$ | 0.30 | 0.60 |

The effect within each stratum is 0.20. Adjustment: $P(Y{=}1\mid do(T{=}1)) = 0.6(0.30)+0.4(0.60)=0.42$ and $P(Y{=}1\mid do(T{=}0))=0.6(0.10)+0.4(0.40)=0.22$, so the ATE (risk difference) is $0.20$. The naive comparison uses $P(z\mid t)$ and gives a larger, biased number because $Z=1$ is over-represented among the treated (§3 computes it).

### 2.3 Continuous covariates: standardization

With continuous $X$ the sum becomes an integral, estimated by the sample average:
$$\hat\psi_t = \frac1n\sum_{i=1}^n \hat m(t, X_i),\qquad \hat m(t,x)=\hat E[Y\mid T=t,X=x].$$
This is **regression adjustment / standardization**, and it is exactly g-computation (Module 19).

### 2.4 Failure mode: positivity

If some $x$ has $P(T=t\mid x)=0$, the term $E[Y\mid T=t,X=x]$ is undefined and the formula silently extrapolates a model (Module 06 §2.3).

## 3. Use It: Code

`code/adjustment_formula_demo.py` computes the exact table values above, the naive conditional, and then confirms both by simulation.

## Exercises

1. Recompute the example with $P(Z=1)=0.8$. Does the adjusted ATE change? Does the naive one?
2. Give an example where the stratum-specific effects differ; what does the adjusted ATE average over?
3. Show that if $Z\perp T$ the adjustment formula equals the naive conditional.

## Key Terms

| Term | What it actually means |
|---|---|
| Adjustment (standardization) formula | $E[Y(t)]=\sum_x E[Y\mid t,x]P(x)$ |
| Exchangeability | $Y(t)\perp T\mid X$ |
| Standardization | Averaging a fitted outcome model over the population covariate distribution |
