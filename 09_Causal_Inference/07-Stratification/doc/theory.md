# 06. Stratification

## Learning Objectives

- Derive the stratified (subclassification) ATE estimator from first principles, using unconfoundedness
- Compute a stratified estimate by hand on a small worked example, and compare it to the naive (unstratified) estimate
- Implement the stratified estimator in code and confirm it removes confounding bias when unconfoundedness holds given the chosen strata
- Recognize the estimator's practical limits, and how they connect back to positivity (Module 06)

---

## 1. The Problem: Turning "Condition on X" Into an Actual Formula

Module 06 stated unconfoundedness as $(Y_i(0),Y_i(1)) \perp T_i \mid X_i$, and Module 06's code demo already previewed a fix: split the data by the confounder $Z$, estimate the effect *within* each group, then combine. This module makes that fix precise, general, and justified — the **stratification** (or subclassification) estimator, one of the most direct ways to translate "conditioning on $X$ solves confounding" into an actual number you can compute.

---

## 2. The Concept

### 2.1 The core idea: compare like with like, within groups

If unconfoundedness holds given $X$, then **within any single value of $X=x$**, treatment is (by assumption) as good as randomly assigned. That means Module 05 §2.3's clean derivation — difference-in-means is unbiased for the ATE — applies *within each stratum defined by $X$*:

$$E\big[Y^{\text{obs}} \mid T=1, X=x\big] - E\big[Y^{\text{obs}}\mid T=0, X=x\big] = E[Y_i(1)-Y_i(0) \mid X_i=x] = \text{CATE}(x)$$

This is just Module 05's difference-in-means logic, applied stratum by stratum instead of to the whole population at once. Each stratum gives you an unbiased estimate of the **conditional** average treatment effect, CATE($x$), for that particular value of $x$ — exactly the quantity Module 05 §2.1 defined.

### 2.2 Averaging across strata to recover the ATE

The overall ATE is the average of CATE($x$) across the population's actual distribution of $X$:

$$\text{ATE} = E\big[\text{CATE}(X_i)\big] = \sum_{x} P(X_i=x) \cdot \text{CATE}(x)$$

Substituting the stratum-level difference-in-means for CATE($x$), and using the sample proportion $\hat{P}(X=x) = n_x/n$ (the fraction of the sample falling into stratum $x$) as an estimate of $P(X=x)$, gives the **stratified ATE estimator**:

$$\widehat{\text{ATE}}_{\text{strat}} = \sum_{x} \frac{n_x}{n} \left(\overline{Y^{\text{obs}}_{T=1, X=x}} - \overline{Y^{\text{obs}}_{T=0,X=x}}\right)$$

In words: compute the naive difference-in-means *separately within each stratum* (where, by assumption, it's unbiased for that stratum's CATE), then take a weighted average across strata, weighting each stratum by how much of the population it represents. This is exactly the computation Module 06's code demo already performed for the binary confounder $Z$ — this lesson generalizes it and justifies it directly from unconfoundedness.

### 2.3 Why this removes confounding bias

Compare the naive (unstratified) estimator to the stratified one directly. The naive estimator implicitly weights each stratum by how *overrepresented it is among the treated* (since it just lumps everyone with $T=1$ together, regardless of $x$) — if a stratum with a systematically higher or lower baseline outcome also happens to be more likely to be treated (Module 01's confounding scenario exactly), that stratum's baseline difference leaks into the naive number as if it were part of the treatment effect. The stratified estimator instead weights each stratum by its **actual size in the population** ($n_x/n$), regardless of how the treated/control split looks within it — so a stratum's baseline outcome level can no longer masquerade as part of the treatment effect, precisely because the comparison never leaves that stratum in the first place.

### 2.4 A complete worked example by hand

Take Module 01's coffee/income table, now with individual-level counts added so we can compute exact stratum-level differences and weights:

| Stratum ($Z$) | $n_z$ (count) | Avg. $Y^{\text{obs}}$, treated ($T=1$) | Avg. $Y^{\text{obs}}$, control ($T=0$) | Stratum diff |
|---|---|---|---|---|
| $Z=1$ (high income) | 1000 | 82.0 | 82.0 | 0.0 |
| $Z=0$ (low income) | 1000 | 74.0 | 74.0 | 0.0 |

(Recall from Module 01 §1.1: the true effect of coffee was exactly zero, with the entire raw gap driven by $Z$.) Applying §2.2's formula:

$$\widehat{\text{ATE}}_{\text{strat}} = \frac{1000}{2000}(82.0-82.0) + \frac{1000}{2000}(74.0-74.0) = \frac{1}{2}(0) + \frac{1}{2}(0) = 0$$

This exactly recovers the true (zero) effect, in sharp contrast to Module 01's naive raw comparison, which found a large, spurious 8-year gap. The stratified estimator succeeded here specifically because $Z$ is the *only* confounder, and we stratified on exactly $Z$ — satisfying unconfoundedness given the chosen stratifying variable.

---

## 3. Practical Limits of Stratification

### 3.1 The curse of dimensionality

Stratification requires enough observations of *both* treatment groups **within every stratum**. With one binary confounder, there are only 2 strata — easy. With five binary confounders, there are $2^5=32$ strata; with several continuous confounders, the number of practically distinct "strata" explodes, and many strata end up with very few (or zero) observations of one treatment group. This connects directly back to Module 06's **positivity** assumption: stratification becomes unreliable, or outright impossible, exactly where positivity is weak or violated, since an empty or near-empty stratum has no meaningful within-stratum comparison to make. (This limitation is the direct motivation for other adjustment methods — propensity score matching, regression adjustment, and more — that exist precisely to handle high-dimensional $X$ more gracefully; a full treatment of those is beyond this crash course.)

### 3.2 Stratification only adjusts for what you stratify on

Just like the unconfoundedness assumption itself (Module 06 §2.2), stratification only removes confounding from the variable(s) you actually stratify on. If some *other*, unmeasured confounder exists, stratifying on $Z$ does nothing to address it — the stratified estimator is only as good as the unconfoundedness assumption it relies on, which remains just as untestable as it was in Module 06.

---

## 4. Use It: Code

`code/stratification_demo.py`:

1. Reproduces §2.4's worked example numerically at a larger sample size, confirming the stratified estimator recovers the true (zero) effect while the naive estimator does not — directly extending Module 01's and Module 06's demos.
2. Extends to a *continuous* confounder by binning it into strata, showing the same bias-removal effect and letting you experiment with how the number of bins affects the result (connecting to §3.1's practical limits — more bins mean more precise stratification but also thinner, noisier strata).
3. Deliberately creates a stratum with very few treated (or untreated) observations, to make §3.1's practical failure mode visible as unstable, high-variance stratum-level estimates, rather than only as an abstract warning.

---

## Exercises

1. Redo §2.4's hand calculation, but this time suppose coffee has a true positive effect of 3 years within *every* income group (so treated averages become 85.0 and 77.0 respectively, keeping the same baselines and weights). Compute $\widehat{\text{ATE}}_{\text{strat}}$ and confirm it now correctly recovers 3.0, unlike the naive raw comparison (which you can also compute, using the population-level treated/control split from Module 01, as a point of comparison).
2. Run `code/stratification_demo.py`'s continuous-confounder section with a small number of bins (e.g., 2) and a large number of bins (e.g., 50). Compare the stratified estimate's accuracy and stability (try rerunning with a different random seed) between the two settings. What tradeoff does this illustrate, and how does it connect to §3.1?
3. Construct your own example (numerically, in code or by hand) with **two** binary confounders instead of one, and compute the stratified ATE estimator across all four resulting strata. Confirm it still recovers the true effect when both confounders are properly stratified on, and confirm that stratifying on only *one* of the two leaves some bias behind.
4. Explain, in your own words, why the stratified estimator is described in this lesson as "only as good as the unconfoundedness assumption it relies on." What would you need to additionally believe (about the *variables you have available in your dataset*) to trust a stratified estimate from a real, non-simulated observational study?

## Key Terms

| Term | What it actually means |
|---|---|
| Stratification / subclassification | Estimating a treatment effect separately within each level of a confounder, then combining these stratum-level estimates into an overall estimate, weighted by stratum size |
| Stratum | A subgroup of the data sharing the same value (or range of values) of the stratifying variable(s) |
| Curse of dimensionality (in stratification) | The rapid growth in the number of strata as more (or more fine-grained) stratifying variables are used, leading to strata with too few observations to estimate reliably |
| Weighted average (in the stratified estimator) | Combining stratum-level effect estimates using each stratum's share of the total population ($n_x/n$) as its weight |
