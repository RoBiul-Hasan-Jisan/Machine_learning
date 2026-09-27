# 08. Confounding

## Learning Objectives

- Recall the informal definition of confounding from Module 01 and restate it precisely
- Distinguish a confounder from other kinds of covariates (mediators, colliders — previewed here, formalized in Module 11)
- Explain confounding bias as a mismatch between an associational quantity and a causal quantity
- See why a graph-based language (Modules 09–16) is needed to handle confounding rigorously once more than one covariate is in play

---

## 1. The Problem: "Adjust for Confounders" Isn't Yet a Precise Instruction

Module 01 introduced confounding informally: a variable $Z$ that affects both treatment $T$ and outcome $Y$, creating a correlation between $T$ and $Y$ that isn't due to a direct causal effect. Module 06 named the assumption this creates a problem for (unconfoundedness) and Module 07 gave one estimator (stratification) that fixes it when $Z$ is the only confounder and is fully observed.

But real problems rarely hand you a single, clearly-labeled confounder. Real datasets have dozens of covariates, and "just adjust for confounders" is not yet a precise instruction until you can answer: *which* covariates, out of everything measured, actually need to be adjusted for? Adjusting for the wrong ones can make things worse, not better — a fact this module previews and Modules 11–16 make fully precise. Getting there requires a more careful, structural definition of confounding than "some third variable causes both" — one built on graphs, which is why this module is the bridge into the DAG machinery of Modules 09–16.

---

## 2. The Concept

### 2.1 Confounding, restated precisely

A variable $Z$ **confounds** the effect of $T$ on $Y$ if it satisfies two conditions simultaneously:

1. $Z$ is a cause of $T$ (or associated with a cause of $T$), **and**
2. $Z$ is a cause of $Y$ through a pathway that does **not** go through $T$.

Both conditions matter. A variable satisfying only (1) — a cause of $T$ with no separate effect on $Y$ — creates no confounding bias, because it has no alternate route to $Y$ to create a spurious association through. A variable satisfying only (2) — a cause of $Y$ unrelated to $T$ — is just an ordinary predictor of the outcome, again with no confounding consequence, because nothing links it back to $T$. It's the **combination** — a common cause with a foot in both camps — that produces the biased comparison Module 01 §1.1 demonstrated numerically.

### 2.2 Confounding bias as a mismatch between two different quantities

Module 04 §2.3 (Interventions) already gave the precise statement of what confounding bias *is*: a gap between the associational quantity $P(Y\mid T=t)$ (or $E[Y\mid T=t]$) and the causal quantity $P(Y\mid do(T=t))$ (or $E[Y\mid do(T=t))]$). When $Z$ confounds the $T\to Y$ relationship, these two quantities differ:

$$E[Y \mid T=t] \ne E[Y \mid do(T=t)]$$

Everything this module and Modules 09–16 build toward is a precise, checkable answer to: **given a specific set of assumptions about how the variables in a problem relate to each other, which covariates, if adjusted for, close this gap?**

### 2.3 A preview: not every covariate should be adjusted for

Here's the fact that makes a graph-based treatment necessary rather than optional. Consider three different roles a covariate $M$ could play relative to $T$ and $Y$:

- **Confounder** ($Z$): $T \leftarrow Z \rightarrow Y$. Must be adjusted for, or the estimate is biased (Module 01).
- **Mediator** ($M$): $T \rightarrow M \rightarrow Y$, i.e., $M$ is a step on the causal pathway from $T$ to $Y$. Adjusting for a pure mediator **removes part of the very effect you're trying to measure** — it's the opposite mistake from ignoring a confounder, but a mistake nonetheless.
- **Collider** ($C$): $T \rightarrow C \leftarrow Y$, i.e., $T$ and $Y$ both cause $C$, with no arrow between $T$ and $Y$'s common cause and $C$ otherwise. Adjusting for a pure collider **creates** a spurious association between $T$ and $Y$ that wasn't there to begin with — a phenomenon called **collider bias**, or Berkson's paradox.

Three structurally different roles, three completely different consequences of "controlling for" the variable — and from a spreadsheet alone, a mediator, a confounder, and a collider can look identical (just another column of numbers correlated with both $T$ and $Y$). Only the causal graph — which variable causes which — tells them apart. This is exactly why Module 09 introduces causal graphs formally, and Module 11 works out chains (mediators), forks (confounders), and colliders as the three fundamental building blocks of every DAG.

### 2.4 A numeric preview of collider bias

To make §2.3's warning about colliders concrete before Module 11 develops it properly, here's a minimal numeric illustration. Suppose $T$ and $Y$ are **completely causally unrelated** — genuinely independent — but both cause a third variable $C$ (say, $T$="won an award," $Y$="has natural talent," $C$="got invited to a TV interview," where interview invitations go out based on either factor):

| | Not invited ($C=0$) | Invited ($C=1$) |
|---|---|---|
| Among people with $T=0$ | most have $Y=0$ | mostly have $Y=1$ (needed something to get invited) |
| Among people with $T=1$ | most have $Y=0$ | some have $Y=1$, but many got invited on $T$ alone |

Restricting attention only to the invited group ($C=1$) — i.e., conditioning on the collider — makes $T$ and $Y$ look *negatively* associated even though they're independent in the general population: among the invited, having $T=1$ makes it *less* necessary to also have $Y=1$ to explain the invitation, and vice versa. This is collider bias, and it can appear the moment you restrict a sample, or "adjust for," any variable that both your treatment and your outcome influence. §3's code demonstrates this numerically alongside a correctly-adjusted confounder, so you can see both effects side by side.

---

## 3. Use It: Code

`code/confounder_vs_collider_demo.py` builds two small simulated datasets side by side:

1. **Confounder case**: $Z \to T$, $Z \to Y$, no direct $T\to Y$ effect (Module 01's setup). Shows the naive (unadjusted) estimate is biased, and adjusting for $Z$ fixes it.
2. **Collider case**: $T$ and $Y$ independent, both cause $C$. Shows the naive (unconditional) estimate correctly finds no association, but *conditioning on $C$* (e.g., restricting the sample to $C=1$) introduces a spurious one — the opposite failure mode from case 1, and the direct illustration of §2.4.

---

## Exercises

1. For each of the following pairs, state whether the third variable named is more likely to be acting as a confounder, a mediator, or a collider, and justify your answer with a plausible causal story: (a) exercise → weight loss, with "diet quality" as the third variable; (b) exercise → weight loss, with "reduced appetite caused by exercise" as the third variable; (c) exercise → weight loss, with "being featured in a health magazine" as the third variable, where being featured requires either visible weight loss or a high public profile from exercising a lot.
2. Run `code/confounder_vs_collider_demo.py` and confirm numerically that adjusting for the confounder removes bias, while conditioning on the collider introduces bias where there was none. Record both estimates (adjusted-confounder and conditioned-collider) alongside their corresponding naive/unconditional baselines.
3. Explain, in one or two sentences, why "adjust for everything you measured" is not a safe default strategy, using the vocabulary (confounder, mediator, collider) introduced in this lesson.
4. In your own words, why is a graph (a DAG) — rather than just a list of variables and their pairwise correlations — necessary to distinguish a confounder from a mediator from a collider?

## Key Terms

| Term | What it actually means |
|---|---|
| Confounder | A variable causally upstream of both treatment and outcome, via a pathway to the outcome that doesn't pass through the treatment |
| Mediator | A variable on the causal pathway from treatment to outcome ($T \to M \to Y$); adjusting for a pure mediator removes part of the true effect |
| Collider | A variable caused by both treatment and outcome ($T \to C \leftarrow Y$); adjusting for (or conditioning on) a pure collider creates spurious association |
| Collider bias / Berkson's paradox | Spurious association between two otherwise-independent variables, induced by conditioning on (or restricting a sample by) their common effect |
