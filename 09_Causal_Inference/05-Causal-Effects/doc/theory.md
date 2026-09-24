# 04. Causal Effects: ATE, ATT, and CATE

## Learning Objectives

- Define ATE, ATT, and CATE precisely, in potential-outcomes notation, and explain when each is the right quantity to ask for
- Prove, using the logic of randomization, why the simple difference-in-means estimator is unbiased for the ATE in a randomized experiment
- Estimate an ATE from simulated RCT data and attach an honest confidence interval to it
- Recognize when ATE, ATT, and CATE can diverge from each other, and why that's a feature, not a bug, of the framework

---

## 1. The Problem: "The" Effect Isn't Always One Number

Module 01 introduced the ATE as a way around the fundamental problem of causal inference: average over individual effects you can't observe individually. But averaging over *which* population, exactly? A drug's average effect across everyone who might ever exist is a different number from its average effect specifically among the people who were actually given it, which is again different from its average effect specifically among, say, people over 65. This module makes those distinctions precise, and — using the DAG/$do(\cdot)$ language from Module 03 — shows exactly why a randomized experiment lets a simple average recover one of these numbers cleanly.

---

## 2. The Concept

### 2.1 Three related, but different, causal quantities

$$\text{ATE} = E[Y_i(1) - Y_i(0)] \qquad \text{(average over the whole population)}$$

$$\text{ATT} = E[Y_i(1) - Y_i(0) \mid T_i = 1] \qquad \text{(average, but only over those who were actually treated)}$$

$$\text{CATE}(x) = E[Y_i(1)-Y_i(0) \mid X_i = x] \qquad \text{(average, within a subgroup defined by covariates } X_i\text{)}$$

**ATE** answers "if we intervened on everyone, what's the average effect?" — useful for population-wide policy decisions. **ATT** (average treatment effect *on the treated*) answers "for the specific people who received treatment, how much did it help them, on average?" — useful when evaluating a program as it was actually deployed, to the population it actually reached, rather than to a hypothetical population that might respond differently. **CATE** answers "what's the effect specifically for people who look like $x$?" — useful for personalizing treatment decisions (Module 19's end-to-end projects, and modern "heterogeneous treatment effect" methods more broadly, build on CATE directly).

### 2.2 Why they can differ: effect heterogeneity

If the treatment effect $\tau_i = Y_i(1)-Y_i(0)$ is the *same* number for every individual (a strong, often unrealistic assumption called constant treatment effects), then ATE = ATT = CATE$(x)$ for every $x$ — there's only one number to talk about. But in general, treatment effects vary across individuals (**effect heterogeneity**), and *who ends up treated* is often correlated with how large their effect would be. Concretely: a job-training program might help people with weaker existing skills much more than people with strong skills; if enrollment is voluntary and weaker-skilled people are more likely to enroll, ATT (averaged over actual enrollees, who skew toward large effects) will be larger than ATE (averaged over everyone, including strong-skilled people who'd benefit little and mostly didn't enroll).

### 2.3 Why a randomized experiment makes ATE easy: no selection into treatment

Recall Module 03 §2.4: in a true RCT, treatment assignment is like a coin flip, causally disconnected from any pre-existing characteristic — formally, $T_i$ is **independent of** $(Y_i(0), Y_i(1))$. This single fact does all the work. Write out the simple difference-in-means estimator and take its expectation:

$$E\big[\,\overline{Y^{\text{obs}}_{T=1}} - \overline{Y^{\text{obs}}_{T=0}}\,\big] = E[Y_i(1)\mid T_i=1] - E[Y_i(0)\mid T_i=0]$$

This uses the fact that among the treated, $Y_i^{\text{obs}}=Y_i(1)$, and among the untreated, $Y_i^{\text{obs}}=Y_i(0)$ (Module 01 §2.3). Now, **because** $T_i$ is independent of $(Y_i(0),Y_i(1))$ under randomization, conditioning on $T_i$ doesn't change these expectations at all:

$$E[Y_i(1) \mid T_i=1] = E[Y_i(1)] \qquad\qquad E[Y_i(0)\mid T_i=0] = E[Y_i(0)]$$

Substituting back in:

$$E\big[\,\overline{Y^{\text{obs}}_{T=1}} - \overline{Y^{\text{obs}}_{T=0}}\,\big] = E[Y_i(1)] - E[Y_i(0)] = \text{ATE}$$

This is the whole proof: the simple difference in observed group means is an **unbiased estimator of the ATE**, precisely because — and only because — randomization broke any link between treatment assignment and potential outcomes. Take away randomization (as in Modules 01 and 03's confounded examples), and the very first equality in this derivation is exactly where things go wrong: $E[Y_i(1)\mid T_i=1] \ne E[Y_i(1)]$ whenever who gets treated depends on their potential outcomes.

### 2.4 Quantifying uncertainty: it's an estimate, not the truth

Even in an RCT, $\overline{Y^{\text{obs}}_{T=1}} - \overline{Y^{\text{obs}}_{T=0}}$ is computed from a finite sample and will vary from one experiment to the next — being unbiased means it's correct *on average across hypothetical repeated experiments*, not that any single run gives exactly the true ATE. The standard error of a difference in two independent sample means is:

$$SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}$$

where $s_1^2, s_0^2$ are the sample variances of $Y^{\text{obs}}$ within the treated and control groups, and $n_1, n_0$ are their sample sizes. A 95% confidence interval is then approximately $\hat{\text{ATE}} \pm 1.96 \times SE$. §3's code computes this alongside the point estimate, so every ATE this module reports comes with an honest sense of how much sampling noise surrounds it.

---

## 3. Use It: Code

`code/ate_estimation_demo.py`:

1. Simulates a true RCT (independent coin-flip treatment assignment, matching §2.3's requirement) with a known, specified true ATE and genuine effect heterogeneity (individual effects vary around that true average, matching §2.2).
2. Computes the simple difference-in-means estimator, plus its standard error and an approximate 95% confidence interval, from §2.4's formula.
3. Separately computes the true ATE, true ATT (by filtering the God's-eye table to the treated group, per Module 02's technique), and a simple CATE (split by a binary covariate) — all directly from the full simulated potential-outcomes table, so you can see numerically how far apart ATE, ATT, and CATE($x$) can be even when they're all computed on the exact same underlying population.

---

## Exercises

1. Follow §2.3's derivation line by line with your own annotations, explaining in words what each equality relies on. Which single fact about randomization is doing all the real work?
2. Run `code/ate_estimation_demo.py` and record the point estimate and 95% CI for the ATE. Increase the sample size by 10x and rerun — what happens to the width of the confidence interval, and does that match what §2.4's $SE$ formula predicts (as $n$ grows, how should $SE$ shrink)?
3. Modify the simulation so that treatment effect heterogeneity is tied to a covariate $X_i$ (e.g., $\tau_i$ is larger when $X_i=1$), and so that people with $X_i=1$ are more likely to be treated (breaking the RCT assumption on purpose). Compute ATE, ATT, and CATE(1)/CATE(0) from the God's-eye table, and confirm ATT ends up closer to CATE(1) than to CATE(0) — explain why, referencing §2.2.
4. Suppose a company wants to decide whether to roll out a discount program to *all* customers. Which of ATE, ATT, or CATE(x) is the right quantity to base that decision on, and why? What if instead they want to decide whether to *continue* a program that's currently only offered to customers who opted in?

## Key Terms

| Term | What it actually means |
|---|---|
| ATE (Average Treatment Effect) | $E[Y_i(1)-Y_i(0)]$, averaged over the whole population |
| ATT (Average Treatment Effect on the Treated) | $E[Y_i(1)-Y_i(0) \mid T_i=1]$, averaged only over those who actually received treatment |
| CATE (Conditional Average Treatment Effect) | $E[Y_i(1)-Y_i(0)\mid X_i=x]$, averaged within a subgroup defined by covariates |
| Effect heterogeneity | The (typically realistic) situation where the treatment effect $\tau_i$ varies across individuals, rather than being one constant number |
| Difference-in-means estimator | $\overline{Y^{\text{obs}}_{T=1}} - \overline{Y^{\text{obs}}_{T=0}}$; unbiased for the ATE specifically when treatment assignment is independent of potential outcomes (as in a true RCT) |
