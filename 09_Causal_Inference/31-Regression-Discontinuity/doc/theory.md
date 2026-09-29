# 31. Regression Discontinuity

## Learning Objectives

- Explain the sharp RD design and its identifying assumption (continuity of potential outcomes at the cutoff)
- Estimate the effect with local linear regression and a kernel, and understand the bandwidth trade-off
- Run the standard validity checks: placebo cutoffs and bandwidth sensitivity
- Distinguish sharp from fuzzy RD and say what each estimates

---

## 1. The Problem

Treatment is often assigned by a rule: scholarship if a test score $\ge 70$, program if income $<$ threshold. Units just above and just below the cutoff are nearly identical, so the cutoff acts like a local experiment.

## 2. The Concept

### 2.1 Sharp RD

Running variable $X$, cutoff $c$, treatment $T=\mathbf 1[X\ge c]$. The identifying assumption is that $E[Y(0)\mid X=x]$ and $E[Y(1)\mid X=x]$ are **continuous in $x$ at $c$**. Then
$$\tau_{RD}=\lim_{x\downarrow c}E[Y\mid X=x]-\lim_{x\uparrow c}E[Y\mid X=x]=E[Y(1)-Y(0)\mid X=c].$$
It is a *local* effect at the cutoff, not an average over all units.

### 2.2 Local linear estimation

Choose bandwidth $h$; fit separate weighted linear regressions of $Y$ on $(X-c)$ on each side using kernel weights $K((X_i-c)/h)$ (e.g. triangular). The RD estimate is the difference between the two fitted intercepts at $c$. A global polynomial fit can be badly biased near the boundary; local linear is preferred.

### 2.3 Bandwidth trade-off

Small $h$: low bias (units are comparable) but few observations, high variance. Large $h$: more data but the linear approximation gets worse. Report results over a range of $h$ and use data-driven selectors (e.g. Imbens–Kalyanaraman, Calonico–Cattaneo–Titiunik).

### 2.4 Validity checks

- **Manipulation:** if units can precisely control $X$ to get just above $c$, continuity fails; check for a jump in the density of $X$ at $c$.
- **Placebo cutoffs:** estimate at cutoffs where there is no treatment; effects should be ~0.
- **Covariate continuity:** pre-treatment covariates should not jump at $c$.

### 2.5 Fuzzy RD

If crossing $c$ only changes the *probability* of treatment, the estimand is the jump in $Y$ divided by the jump in $T$, an IV estimator (Module 30) with $\mathbf 1[X\ge c]$ as instrument, identifying an effect for compliers at the cutoff.

## 3. Use It: Code

`code/rdd_demo.py` simulates a nonlinear $E[Y\mid X]$ with a true jump of 2 at $c=0$, and compares the naive treated–control difference, local linear RD across bandwidths, and a placebo cutoff.

## Exercises

1. Add manipulation (units near $c$ push themselves over). What happens to the estimate?
2. Replace local linear with a global quadratic. Compare.
3. Why is the RD effect not generalizable to units far from the cutoff?

## Key Terms

| Term | What it actually means |
|---|---|
| Running variable | The score that determines treatment through a cutoff |
| Bandwidth | Window around the cutoff used for estimation |
| Local linear regression | Kernel-weighted linear fit on each side of the cutoff |
| Fuzzy RD | Cutoff changes treatment probability rather than deterministically |
