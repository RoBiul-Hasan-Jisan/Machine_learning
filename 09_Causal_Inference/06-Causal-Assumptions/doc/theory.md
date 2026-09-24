# 05. Causal Assumptions

## Learning Objectives

- State the four core assumptions (SUTVA, unconfoundedness/ignorability, positivity/overlap, consistency) precisely, in potential-outcomes and/or DAG notation
- Explain, for each assumption, a concrete scenario where it is violated and what goes wrong numerically as a result
- Distinguish assumptions that are testable from data from those that are fundamentally untestable, and explain why
- Recognize unconfoundedness as the assumption that makes observational (non-randomized) causal estimation possible at all

---

## 1. The Problem: Randomization Isn't Always Available

Module 05 showed that a true RCT makes ATE estimation clean, because randomization guarantees $T_i \perp (Y_i(0), Y_i(1))$ — treatment is independent of potential outcomes. But most data available in the world is **observational**: nobody flipped a coin to decide who smokes, who gets a promotion, or who adopts a new technology. This module collects the assumptions that must hold for observational data to still support a valid causal estimate — and is explicit about the fact that some of these assumptions can be checked against data, while others fundamentally cannot be, and must instead be argued for using domain knowledge.

---

## 2. The Concept

### 2.1 SUTVA — already introduced, restated as an assumption

Module 02 §2.4 introduced SUTVA (Stable Unit Treatment Value Assumption): no interference between units, and no hidden variation within a treatment level. It's listed here again because it belongs alongside the other three as a *precondition* — without it, $Y_i(1)$ isn't even well-defined as a single number, so none of the other assumptions or estimators in this crash course are meaningful yet.

### 2.2 Unconfoundedness / Ignorability: the central assumption for observational data

$$\big(Y_i(0), Y_i(1)\big) \perp T_i \;\Big|\; X_i$$

Read this as: "once you know an individual's covariates $X_i$ (age, income, prior health, whatever relevant characteristics you've measured), their treatment assignment $T_i$ tells you nothing further about their potential outcomes." Equivalently, using Module 03's DAG language: **all backdoor paths from $T$ to $Y$ are blocked once you condition on $X$** — there's no *unmeasured* confounder like Module 01's $Z$ still lurking in the background. This is often just called **ignorability**, since it says treatment assignment can be "ignored" (treated as if random) once $X$ is accounted for.

**Why this can't be tested from data alone:** unconfoundedness is a claim about $Y_i(0)$ and $Y_i(1)$ jointly, but (Module 01's fundamental problem) you never observe both for the same person — so you can never directly check whether treatment assignment really is independent of the *pair*. You can check whether measured covariates $X$ look balanced between treated and control groups (a common diagnostic), but that says nothing about *unmeasured* confounders, which is exactly what unconfoundedness needs to rule out. This is why unconfoundedness is typically defended with a substantive argument ("we measured every plausible common cause of $T$ and $Y$ that domain experts could think of") rather than a statistical test — it is fundamentally an assumption, not a fact you derive from the data in front of you.

### 2.3 Positivity / Overlap: everyone needs a chance at both treatments

$$0 < P(T_i=1 \mid X_i=x) < 1 \qquad \text{for every value } x \text{ that actually occurs}$$

This says: for every combination of covariates seen in the data, there must be a genuine chance of observing *both* treated and untreated individuals. If everyone with $X_i = x$ was, without exception, treated, there's no one similar (in terms of $X$) to compare them to — any estimate for that subgroup would be pure extrapolation, not something grounded in comparable observed data. **Unlike unconfoundedness, positivity is checkable from data**: for each region of covariate space, you can directly count (or estimate the probability of) treated and untreated individuals present, and flag regions with none of one group.

A useful way to picture a violation: if only people under 40 ever receive a particular surgery in your dataset, you have no data-driven way to estimate the surgery's effect for people over 40 — extending an estimate there requires assuming the effect generalizes, which is a modeling choice, not something the data itself supports.

### 2.4 Consistency: "treatment" means what you think it means

$$T_i = t \implies Y_i^{\text{obs}} = Y_i(t)$$

This says: the potential outcome under the treatment level someone actually received is, in fact, what got observed — there's no gap between "the treatment as defined in the potential-outcomes framework" and "the treatment as it was actually administered in practice." This sounds almost too obvious to state, but it's closely related to SUTVA's "no hidden variation" clause (§2.1): if "treatment" nominally means "attended the training program," but attendance quality varied wildly (some people attended one session, others attended all ten), then $Y_i(1)$ isn't a single well-defined quantity, and consistency quietly fails even though everyone who was "treated" really was treated in some loose sense.

### 2.5 Summary table: testable or not?

| Assumption | Can it be checked directly against data? | What breaks if it's violated |
|---|---|---|
| SUTVar (SUTVA) | Partially — interference can sometimes be detected via study design, but not always from a single dataset | $Y_i(1)$/$Y_i(0)$ aren't even well-defined single numbers |
| Unconfoundedness | **No** — fundamentally untestable, since it's a joint claim about both potential outcomes | Estimates are biased by unmeasured confounding (Module 01's $Z$) |
| Positivity/overlap | **Yes** — directly observable by examining the covariate distribution within treatment groups | Estimates for under-covered regions of $X$ become pure extrapolation |
| Consistency | Partially — depends on how precisely "treatment" was defined and measured | The estimated effect corresponds to no single well-defined intervention |

The bolded row (unconfoundedness) is worth sitting with: it is the assumption doing the most work in observational causal inference, and it is also the one you can never fully verify. This is precisely why domain expertise, careful data collection (measuring every plausible confounder), and sensitivity analysis (checking how much a hidden confounder *would* have to matter to overturn your conclusion — beyond this crash course's scope, but a standard next step) are as central to causal inference as any estimator formula.

---

## 3. Use It: Code

`code/assumptions_violation_demo.py` demonstrates each assumption's violation numerically, side by side with a version where the assumption holds, using the same confounded/RCT simulation machinery from Modules 01–04:

1. **Unconfoundedness violated**: reruns Module 01's confounding demo, showing the naive estimate stays biased however much data you collect, because the missing confounder $Z$ is never measured or conditioned on.
2. **Unconfoundedness restored by conditioning on the right $X$**: adjusts the same simulation to instead condition on the confounder (a preview of Module 07's stratification), showing the bias resolves once the relevant $X$ is properly accounted for.
3. **Positivity violated**: constructs a covariate region where every observation is treated (no untreated comparison group exists there at all) and shows what happens when you naively try to estimate an effect for that region anyway.

---

## Exercises

1. For each of SUTVA, unconfoundedness, positivity, and consistency, write one sentence stating what would have to be true about a *specific* real-world study (choose your own: a marketing campaign, a medical treatment, an educational intervention) for that assumption to hold.
2. Explain, in your own words, why unconfoundedness is untestable but positivity is testable. What's the structural difference between the two assumptions that causes this?
3. Run `code/assumptions_violation_demo.py`'s unconfoundedness section. Then modify it to increase the sample size by 100x. Does the bias in the naive estimate shrink as sample size grows? What does this tell you about the difference between a bias problem and a sampling-noise (variance) problem — and why does Module 05's confidence interval (a variance concept) not help with a violated-unconfoundedness problem?
4. Run the positivity-violation section. Try to compute a naive estimate for the region with no untreated comparisons anyway (e.g., by fitting a simple linear model and extrapolating). Compare this "estimate" to a region where both groups are present. What specifically makes the first number less trustworthy, even though the code will happily compute *some* number either way?

## Key Terms

| Term | What it actually means |
|---|---|
| Unconfoundedness / Ignorability | $(Y_i(0),Y_i(1)) \perp T_i \mid X_i$ — treatment assignment is as good as random once covariates $X_i$ are accounted for; fundamentally untestable from data alone |
| Positivity / Overlap | Every covariate value that occurs in the data must have a genuine chance of appearing in both the treated and untreated groups; checkable directly from data |
| Consistency | The observed outcome under an individual's actual treatment equals their potential outcome for that treatment level — "treatment" is well-defined and consistently administered |
| Extrapolation (in the positivity context) | Estimating an effect for a region of covariate space with no comparable observed data, relying entirely on model assumptions rather than direct comparison |
