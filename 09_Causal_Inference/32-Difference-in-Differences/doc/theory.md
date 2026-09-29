# 32. Difference-in-Differences

## Learning Objectives

- Derive the 2×2 DiD estimator and the parallel trends assumption
- Explain why neither a simple before/after nor a simple treated/control comparison works
- Use an event-study plot (coefficients by period) to check pre-trends
- Know the main threats: violated parallel trends, anticipation, and biases of two-way fixed effects with staggered adoption

---

## 1. The Problem

A policy hits one region in a given year. Comparing that region before/after confuses the policy with the time trend. Comparing it with another region confuses the policy with permanent differences between regions. **DiD** uses both comparisons to cancel each nuisance.

## 2. The Concept

### 2.1 The estimator

Groups $g\in\{\text{treated},\text{control}\}$, periods pre and post:
$$\widehat{DiD}=\big(\bar Y_{T,post}-\bar Y_{T,pre}\big)-\big(\bar Y_{C,post}-\bar Y_{C,pre}\big).$$

### 2.2 Identification

Assume **parallel trends**: absent treatment, the treated group's average outcome would have changed by the same amount as the control group's,
$$E[Y_{post}(0)-Y_{pre}(0)\mid T{=}1]=E[Y_{post}(0)-Y_{pre}(0)\mid T{=}0].$$
Also **no anticipation** (outcomes before treatment are not affected) and SUTVA. Then $\widehat{DiD}$ estimates the **ATT**:
$$E[Y_{post}(1)-Y_{post}(0)\mid T{=}1].$$
Levels may differ across groups; only *trends* must match.

### 2.3 Event study

With several periods, plot $\hat\delta_t=(\bar Y_{T,t}-\bar Y_{C,t})-(\bar Y_{T,t_0}-\bar Y_{C,t_0})$ relative to a reference period $t_0$. Pre-treatment coefficients near 0 support parallel trends (they do not *prove* it, and pre-testing has its own issues).

### 2.4 Threats

- **Diverging trends:** if the treated group was already trending differently, DiD absorbs that difference into the "effect".
- **Staggered adoption:** with units treated at different times, two-way fixed-effects regressions can put negative weights on some comparisons (Goodman-Bacon 2021); modern estimators (Callaway–Sant'Anna, Sun–Abraham) avoid this.
- **Composition changes** and spillovers between groups.

## 3. Use It: Code

`code/did_demo.py` simulates 6 periods with treatment starting at $t=3$ (true effect 2), a common time trend and level differences. It compares naive estimators with DiD, prints event-study coefficients, and repeats with a differential trend to show the bias.

## Exercises

1. Shift the treatment date earlier in the data-generating process but not in the analysis (anticipation). What happens to the pre-period coefficients?
2. Compute DiD using only periods 2 and 3. Compare with the multi-period version.
3. Why does DiD estimate the ATT and not the ATE?

## Key Terms

| Term | What it actually means |
|---|---|
| Parallel trends | Untreated potential outcomes would have moved in parallel across groups |
| Event study | Period-by-period treated–control gaps relative to a reference period |
| Anticipation | Behavior changes before treatment starts |
