# 01. Introduction to Causal Effect

## Learning Objectives

- Explain precisely why correlation does not imply causation, using a confounding example
- State the "fundamental problem of causal inference" and why it makes causal questions different from ordinary prediction questions
- Introduce potential-outcomes notation, $Y_i(1)$ and $Y_i(0)$, and read a definition written with it
- See the roadmap for the rest of this crash course

---

## 1. The Problem: Two Different Questions That Look Alike

Suppose you notice that people who drink more coffee tend to live longer. Two very different explanations are consistent with that observation:

1. Coffee causes longer life (a **causal** claim).
2. Something else — say, having enough money and free time to enjoy coffee, and to also afford healthcare, exercise, and better food — makes people both drink more coffee *and* live longer (coffee and longevity are correlated only because they share a common cause).

A predictive model (a regression, a neural network from earlier modules in this curriculum) is very good at answering "if I observe someone drinking a lot of coffee, what's my best guess about their lifespan?" It is not, by itself, able to answer "if I *made* someone drink more coffee, would their lifespan change?" These are different questions, and confusing them is the single most common error in applied data analysis. This whole crash course is about making that distinction precise, and about the machinery needed to answer the second kind of question — the causal kind — rigorously.

### 1.1 A concrete numeric example of confounding

Imagine a (simplified, exaggerated) population split into two groups by free time and income, call it $Z$:

| Group ($Z$) | Fraction of population | Avg. cups of coffee/day | Avg. lifespan (years) |
|---|---|---|---|
| High free time/income | 50% | 4 | 82 |
| Low free time/income | 50% | 1 | 74 |

Averaging over the whole population, people who drink 4 cups live longer (82) than people who drink 1 cup (74) — a strong *correlation*. But suppose, within each group, coffee itself has **zero** true effect on lifespan; the entire 8-year gap is driven by $Z$ (income/free time), which independently pushes both coffee consumption and lifespan up. If you intervened and forced the low-income group to drink 4 cups a day, their lifespan would very plausibly stay at 74 — because $Z$, not coffee, is doing all the work. The correlation is real; the causal claim built on top of it, that coffee extends life, is not. §2 below and the code in this module simulate exactly this scenario so you can see the gap between the two numbers directly.

---

## 2. The Concept

### 2.1 Potential outcomes: imagining the road not taken

The precise language causal inference uses to separate "what happened" from "what caused it" is the **potential outcomes** framework (also called the Neyman-Rubin causal model). For each individual $i$ and a binary treatment (coffee: yes/no; a drug: given/not given; a policy: enacted/not), define two numbers:

$$Y_i(1) = \text{the outcome individual } i \text{ would have, if given the treatment}$$
$$Y_i(0) = \text{the outcome individual } i \text{ would have, if } \textbf{not} \text{ given the treatment}$$

Both of these are defined for *every* individual, regardless of what actually happened to them. $Y_i(1)$ and $Y_i(0)$ are properties of the individual and the world, not of what was observed. This is the key conceptual move of the whole field: a causal effect is a comparison between two potential outcomes for the *same* individual, e.g. $Y_i(1) - Y_i(0)$ — not a comparison between two *different* individuals' observed outcomes, which is all a naive correlation can offer.

### 2.2 The fundamental problem of causal inference

Here is the catch that makes causal inference genuinely hard, not just a notational exercise: for any individual $i$, you can only ever **observe one** of $Y_i(1)$ or $Y_i(0)$ — whichever potential outcome corresponds to the treatment they actually received. If Alice actually drank coffee, you observe $Y_{\text{Alice}}(1)$; you never get to see $Y_{\text{Alice}}(0)$, the lifespan she *would have had* had she not been a coffee drinker, because that world didn't happen. This is called the **fundamental problem of causal inference**: individual-level causal effects are, in general, permanently unobservable. Everything in this crash course — interventions (Module 03), causal effects at the population level (Module 05), the assumptions that let us proceed anyway (Module 06), and stratification (Module 07) — exists because of this one problem, and is about finding principled, honest ways to say something useful *despite* it.

### 2.3 The observed outcome, written in potential-outcomes notation

Let $T_i \in \{0, 1\}$ denote whether individual $i$ actually received the treatment. What's actually observed is:

$$Y_i^{\text{obs}} = T_i \cdot Y_i(1) + (1-T_i) \cdot Y_i(0)$$

Read this literally: if $T_i = 1$ (treated), the formula picks out $Y_i(1)$ and zeroes out $Y_i(0)$'s term; if $T_i=0$, the reverse. This equation is the formal version of §2.2's fundamental problem — it makes explicit that the data you actually collect is always a *mixture*, determined by treatment assignment, of two underlying (mostly unobserved) potential outcomes.

### 2.4 Individual vs. average treatment effects

The **individual treatment effect** for person $i$ is $\tau_i = Y_i(1) - Y_i(0)$ — by §2.2, this is unobservable for essentially everyone, essentially always. What *can*, under the right conditions, be estimated is a population-level summary, most commonly the **average treatment effect**:

$$\text{ATE} = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]$$

Module 05 works through exactly how this quantity is estimated from real data, and Module 06 works through exactly which assumptions make that estimation valid rather than just numerically possible.

---

## 3. Where This Crash Course Goes From Here

| Module | What it covers |
|---|---|
| 01 (this one) | Why correlation ≠ causation, potential outcomes notation, the fundamental problem of causal inference |
| 02 | Potential outcomes and counterfactuals in depth — the "God's-eye view" simulation trick used throughout this course |
| 03 | Interventions: the $do(\cdot)$ operator, and why $P(Y \mid do(X))$ is a fundamentally different quantity from $P(Y \mid X)$ |
| 04 | Hypothetical interventions: individual-level counterfactual queries via the abduction-action-prediction procedure, going beyond Module 03's population-level $do(\cdot)$ |
| 05 | Causal effects: ATE, ATT, CATE, and how randomized experiments let you estimate them from observed data alone |
| 06 | Causal assumptions: unconfoundedness/ignorability, SUTVA, positivity/overlap, consistency — what has to be true for an estimate to mean what you want it to mean |
| 07 | Stratification: a concrete estimator that adjusts for confounding by conditioning on strata of a confounder, built directly on Module 06's assumptions |

---

## Exercises

1. Come up with your own example (not coffee/lifespan) of two variables that are correlated but where you suspect a confounder, rather than a direct causal effect, explains the correlation. Name the plausible confounder explicitly.
2. Using §2.3's equation $Y_i^{\text{obs}} = T_i Y_i(1) + (1-T_i)Y_i(0)$, write out what $Y_i^{\text{obs}}$ equals in the two cases $T_i=1$ and $T_i=0$ separately, and confirm both match what you'd intuitively expect.
3. In your own words, explain why the fundamental problem of causal inference (§2.2) does *not* say that causal questions are impossible to answer — only that they can't be answered by looking at *one individual alone*. What kind of comparison (hint: think about groups of people) might get around this, at least approximately? (Module 05 answers this properly, but take a guess first.)
4. For the confounding example in §1.1, suppose instead that within each income group, coffee genuinely does add 3 years of life. Recompute the average lifespan for the "4 cups/day" and "1 cup/day" groups under this new assumption, keeping the group-level baseline lifespans (82, 74) as still being driven partly by $Z$. Is the raw group difference in average lifespan now equal to, greater than, or less than the true causal effect of 3 years? What does this tell you about reading a raw correlation as if it were a causal effect size?

## Key Terms

| Term | What it actually means |
|---|---|
| Confounder | A variable that influences both the treatment and the outcome, creating a correlation between them that isn't due to a direct causal effect |
| Potential outcomes, $Y_i(1)$, $Y_i(0)$ | The outcome individual $i$ *would have* under treatment, or under no treatment, respectively — both defined regardless of which was actually observed |
| Fundamental problem of causal inference | The fact that only one of $Y_i(1)$, $Y_i(0)$ is ever observed for any given individual, making individual causal effects permanently unobservable |
| Individual treatment effect ($\tau_i$) | $Y_i(1) - Y_i(0)$ for a specific individual — unobservable in practice |
| Average treatment effect (ATE) | $E[Y_i(1)] - E[Y_i(0)]$, a population-level summary that can, under the right assumptions, be estimated from data |
