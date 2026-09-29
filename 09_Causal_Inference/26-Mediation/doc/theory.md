# 26. Mediation

## Learning Objectives

- Define natural direct and indirect effects with nested potential outcomes
- State the sequential ignorability assumptions and identify the "cross-world" one that cannot be tested
- Recover direct and indirect effects in the linear case (product of coefficients)
- Show how an unmeasured mediator–outcome confounder biases the decomposition even in a randomized trial

---

## 1. The Problem

Module 08 warned that adjusting for a mediator removes part of the effect. But often the **mechanism** is the question: how much of the effect of $T$ on $Y$ goes *through* $M$, and how much does not?

## 2. The Concept

### 2.1 Nested potential outcomes

Let $M(t)$ be the mediator under treatment $t$ and $Y(t,m)$ the outcome under treatment $t$ and mediator value $m$.

$$TE=E[Y(1,M(1))-Y(0,M(0))]=\underbrace{E[Y(1,M(0))-Y(0,M(0))]}_{NDE}+\underbrace{E[Y(1,M(1))-Y(1,M(0))]}_{NIE}.$$

- **NDE (natural direct effect):** change $T$ but hold the mediator at the value it would have had *without* treatment.
- **NIE (natural indirect effect):** keep treatment at 1 but change the mediator from its untreated to its treated value.

### 2.2 Identification (sequential ignorability)

Given covariates $X$: (i) $Y(t,m)\perp T\mid X$, (ii) $M(t)\perp T\mid X$, (iii) $Y(t,m)\perp M\mid T,X$ (**no mediator–outcome confounding**), and (iv) $Y(t,m)\perp M(t')\mid X$ (the **cross-world** assumption). Assumption (iv) concerns the joint distribution of $Y(1,\cdot)$ and $M(0)$, which can never be observed in the same person, so it is untestable even in a randomized trial.

### 2.3 Linear case: product of coefficients

If $M=\alpha T+\varepsilon_M$ and $Y=cT+bM+\varepsilon_Y$ with no confounding of $M\to Y$:
$$NDE=c,\qquad NIE=\alpha b,\qquad TE=c+\alpha b.$$
Estimate $\alpha$ from $M\sim T$, and $c,b$ from $Y\sim T+M$.

### 2.4 Mediator–outcome confounding

Randomizing $T$ does not randomize $M$. If an unmeasured $U$ affects both $M$ and $Y$, the coefficient $b$ is biased, so both NDE and NIE estimates are wrong even though the **total effect** is still fine. Adjusting for post-treatment variables also breaks the causal interpretation when a confounder of $M$ and $Y$ is itself affected by $T$.

## 3. Use It: Code

`code/mediation_demo.py` simulates a randomized $T$ with $\alpha=0.8$, $b=1.5$, $c=0.5$ (true TE $=1.7$, NDE $=0.5$, NIE $=1.2$). It estimates the decomposition without and with an unmeasured mediator–outcome confounder, and shows that adjusting for $M$ gives only the direct effect.

## Exercises

1. Add a $T\times M$ interaction to $Y$. How do NDE and NIE now depend on the reference level?
2. Increase the strength of $U\to M$ and $U\to Y$. How does bias in NIE scale?
3. Why is (iv) untestable even in an RCT?

## Key Terms

| Term | What it actually means |
|---|---|
| NDE | Effect of $T$ with the mediator held at its untreated value |
| NIE | Effect of shifting the mediator as treatment would, treatment held fixed |
| Cross-world assumption | Independence involving potential outcomes under different treatment levels |
