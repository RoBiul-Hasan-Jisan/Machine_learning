# 28. Selection Bias

## Learning Objectives

- Distinguish confounding (a bias in *who is treated*) from selection bias (a bias in *who is in the data*)
- Show how selection on a collider biases even a randomized comparison
- Use inverse probability of selection weighting (IPSW) when selection depends on observed covariates
- Recognize common forms: loss to follow-up, survivorship, volunteer bias, missing outcomes

---

## 1. The Problem

Everything so far assumed the analysis sample is representative. **Selection bias** arises when the units we observe were chosen in a way related to treatment and outcome.

## 2. The Concept

### 2.1 Two structures

Let $S=1$ mean "included in the analysis".

- **Selection on a collider (internal validity):** $T\to S\leftarrow Y$ (or $S$ caused by a descendant of $T$ and something affecting $Y$). Analyzing only $S=1$ conditions on a collider, creating association between $T$ and $Y$ even if $T$ was randomized. Examples: loss to follow-up related to treatment side effects and outcome; studying only survivors.
- **Selection on effect modifiers (external validity):** $S$ depends on covariates $X$ that modify the treatment effect. The estimate is internally valid for the selected population but differs from the target population's effect.

### 2.2 Why randomization does not save you

Randomization guarantees $T\perp Y(t)$ in the *population*. It says nothing about $T\perp Y(t)\mid S=1$ if $S$ is affected by $T$ and by causes of $Y$.

### 2.3 Inverse probability of selection weighting

If $S\perp (Y(0),Y(1))\mid X$ and $P(S=1\mid X)>0$, weighting the selected units by $1/P(S=1\mid X)$ re-creates the target population:
$$\widehat{ATE}_{pop}=\text{weighted diff-in-means with } w_i=\frac{1}{\hat P(S_i=1\mid X_i)}.$$
This needs $X$ measured on **both** selected and non-selected units (or known population margins).

### 2.4 What cannot be fixed by weighting

If selection depends on the outcome itself (or on unmeasured factors that affect the outcome), weighting on observed covariates cannot repair it. Sensitivity analyses or additional data are needed (Module 34).

## 3. Use It: Code

`code/selection_bias_demo.py` shows (A) a randomized $T$ with selection depending on $T$ and $Y$, where the selected-sample effect is biased, and (B) selection depending on an effect modifier, where IPSW recovers the population ATE.

## Exercises

1. In (A), vary how strongly $S$ depends on $Y$. When does the bias vanish?
2. In (B), use a misspecified selection model. What happens?
3. Distinguish "loss to follow-up" bias from confounding using a DAG.

## Key Terms

| Term | What it actually means |
|---|---|
| Selection bias | Bias from analyzing a non-representative or collider-conditioned subsample |
| IPSW | Weighting selected units by the inverse probability of being selected |
| Survivorship bias | Selection on survival, which is affected by exposure and outcome determinants |
