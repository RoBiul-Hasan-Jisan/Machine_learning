# 30. Instrumental Variables

## Learning Objectives

- State the three IV conditions (relevance, independence, exclusion) and read them off a DAG
- Derive the Wald estimator and 2SLS
- Explain what a binary-instrument IV estimates (the LATE, for compliers) and the monotonicity assumption
- Show how a weak instrument makes IV unreliable

---

## 1. The Problem

With an unmeasured confounder $U$, adjustment fails (Module 17). An **instrument** $Z$ is a source of variation in $T$ that is unrelated to $U$ and affects $Y$ only through $T$.

## 2. The Concept

### 2.1 Conditions

DAG: $Z\to T\to Y$, $U\to T$, $U\to Y$.

1. **Relevance:** $Z$ affects $T$ ($\text{Cov}(Z,T)\ne0$).
2. **Independence:** $Z\perp U$ (as if randomized, possibly conditional on covariates).
3. **Exclusion:** $Z$ affects $Y$ only through $T$.

Only relevance is testable.

### 2.2 Wald estimator and 2SLS

For $Y=\beta T+\gamma U+\varepsilon$, $\text{Cov}(Z,Y)=\beta\,\text{Cov}(Z,T)$, so
$$\hat\beta_{IV}=\frac{\text{Cov}(Z,Y)}{\text{Cov}(Z,T)}=\frac{\text{reduced form}}{\text{first stage}}.$$
2SLS: regress $T$ on $Z$ (and covariates) to get $\hat T$, then regress $Y$ on $\hat T$. With one instrument it equals the Wald ratio.

### 2.3 LATE with a binary instrument

Compliers take treatment only when encouraged. Assume **monotonicity** (no defiers). Then
$$\frac{E[Y\mid Z{=}1]-E[Y\mid Z{=}0]}{E[T\mid Z{=}1]-E[T\mid Z{=}0]}=E[Y(1)-Y(0)\mid \text{compliers}]=LATE.$$
It is the effect for compliers only, not necessarily the ATE.

### 2.4 Weak instruments

When the first stage is weak, the ratio's denominator is close to zero, so $\hat\beta_{IV}$ has extremely heavy tails (in §3, a standard deviation of about 59 for a true effect of 1, so means over repeated samples are meaningless and medians are more informative), and in finite samples it is biased toward OLS. Report the first-stage $F$; a common rule of thumb is $F>10$.

## 3. Use It: Code

`code/iv_demo.py` (i) compares OLS, Wald/2SLS with a strong instrument and a weak one across repeated samples; (ii) simulates always-takers, never-takers and compliers with different effects and shows the Wald ratio recovers the complier effect.

## Exercises

1. Violate exclusion by adding $Z\to Y$. What happens?
2. Compute the first-stage $F$ for the weak instrument.
3. Why does IV not identify the ATE unless effects are homogeneous?

## Key Terms

| Term | What it actually means |
|---|---|
| Instrument | Affects treatment, is independent of confounders, affects outcome only via treatment |
| LATE | Average effect among compliers |
| Weak instrument | Small first-stage association; makes IV noisy and biased |
