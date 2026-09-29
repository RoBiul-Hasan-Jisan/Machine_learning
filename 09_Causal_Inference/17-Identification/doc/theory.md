# 17. Identification

## Learning Objectives

- Define identifiability formally, and state what it does and does not promise
- Show, with an explicit counterexample, that a causal effect can be **non-identified**: two different causal models produce exactly the same observed data but different causal effects
- Distinguish identification from estimation, and from the assumptions (Module 06) that make identification possible
- Name the main identification strategies covered in the rest of Part 3

---

## 1. The Problem: Can the Data Even Answer the Question?

Module 16 introduced identification informally as the step that comes *before* estimation. This module makes it formal. Everything in Modules 18–37 is a different strategy for answering one question: **given assumptions $\mathcal{A}$ about how the data were generated, can the causal quantity be written purely in terms of the observed distribution?** If yes, it is *identified* and estimation is meaningful. If no, no amount of data will help.

---

## 2. The Concept

### 2.1 Formal definition

Let $\mathcal{M}$ be the set of all causal models consistent with assumptions $\mathcal{A}$, and let $P_M$ be the observed-data distribution that model $M$ generates. A causal quantity $\psi(M)$ (for example, $E[Y\mid do(T=1)] - E[Y \mid do(T=0)]$) is **identified** if:

$$P_{M_1} = P_{M_2} \;\Longrightarrow\; \psi(M_1) = \psi(M_2) \quad \text{for all } M_1, M_2 \in \mathcal{M}.$$

In words: if two models could ever be confused with each other from data alone, they must agree on the causal quantity. If two models generate *identical* observed distributions but *different* causal effects, the effect is **not identified** under $\mathcal{A}$.

### 2.2 A concrete non-identification example

Consider two linear-Gaussian models for $(T, Y)$, both with $\operatorname{Var}(T)=1$:

- **Model A** (no confounding): $T \sim N(0,1)$, $\;Y = 0.6\,T + \varepsilon_Y$, $\;\varepsilon_Y\sim N(0, 2)$. True causal effect = **0.6**.
- **Model B** (pure confounding): $U\sim N(0,1)$, $\;T = 0.5\,U + \varepsilon_T$ with $\operatorname{Var}(\varepsilon_T)=0.75$, $\;Y = 1.2\,U + \varepsilon_Y$ with $\operatorname{Var}(\varepsilon_Y)=0.92$. Here $T$ has **no** effect on $Y$. True causal effect = **0**.

Check the observed covariance: in Model A, $\operatorname{Cov}(T,Y)=0.6$ and $\operatorname{Var}(Y)=0.36+2=2.36$. In Model B, $\operatorname{Cov}(T,Y)=0.5\cdot1.2=0.6$ and $\operatorname{Var}(Y)=1.44+0.92=2.36$. Because both models are jointly Gaussian with the same mean and covariance, their observed distributions are **identical**, yet one has an effect of 0.6 and the other 0. With $U$ unmeasured, the data cannot tell them apart. The effect is not identified without further assumptions — for instance, "no unmeasured confounding" (Module 06), which would rule out Model B.

### 2.3 What identification assumptions look like

Identification always comes from assumptions that shrink $\mathcal{M}$ until all remaining models agree on $\psi$:

| Strategy | Key assumption | Module |
|---|---|---|
| Adjustment / g-computation / IPW | No unmeasured confounding given $X$ + positivity | 18–24 |
| Front-door | A mediator fully carries the effect and is unconfounded with $T$ | 36 |
| Instrumental variables | A valid instrument (relevance, exclusion, independence) | 30 |
| RDD | Continuity of potential outcomes at a cutoff | 31 |
| Difference-in-differences | Parallel trends | 32 |
| Synthetic control | Treated unit's counterfactual is a weighted donor combination | 33 |
| Sensitivity analysis | Bounds on how strong hidden confounding could be | 34 |

### 2.4 Identification is not estimation, and not truth

An identified quantity can still be estimated badly (small samples, misspecified models), and an identification argument is only as credible as its assumptions — which are usually untestable. Identification tells you *what you would need to believe*, not whether you should believe it.

---

## 3. Use It: Code

`code/non_identification_demo.py` simulates Models A and B, shows their empirical covariance matrices are indistinguishable, computes each model's true causal effect by simulating $do(T=t)$, and shows that adding "observing $U$" makes the effect identified again.

---

## Exercises

1. Explain in one paragraph why "same observed distribution, different causal effect" implies that no estimator, however clever, can recover the effect.
2. Modify Model B so that $T$ has a true effect of 0.3 and re-tune the other parameters so the observed covariance still matches Model A. Is that always possible?
3. Which assumption from Module 06 rules out Model B, and what would you have to measure to check it?
4. Give an example from your own field where an effect is plausibly non-identified without an instrument or experiment.

## Key Terms

| Term | What it actually means |
|---|---|
| Identifiability | A causal quantity is identified if it is uniquely determined by the observed distribution under the stated assumptions |
| Non-identification | Distinct causal models consistent with the assumptions generate the same observed data but different causal effects |
| Identification strategy | A set of assumptions plus a formula that expresses a causal quantity in terms of observable distributions |
