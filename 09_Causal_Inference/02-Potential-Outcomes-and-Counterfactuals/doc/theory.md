# 02. Potential Outcomes and Counterfactuals

## Learning Objectives

- Distinguish a "potential outcome" from a "counterfactual" precisely
- Build and read a full potential-outcomes table (the "God's-eye view") for a small population
- Compute individual treatment effects and the ATE from a God's-eye table, and see exactly which numbers a real study could never access
- Explain SUTVA (the Stable Unit Treatment Value Assumption) and why it's silently required just to write $Y_i(1)$ as a single number

---

## 1. The Problem: We Need to Reason About Worlds We Can't See

Module 01 introduced $Y_i(1)$ and $Y_i(0)$ and stated the fundamental problem: only one is ever observed per person. This module treats potential outcomes as objects worth studying carefully in their own right — including a simulation trick (the "God's-eye view") that lets us, as people building and studying the framework, temporarily pretend we can see both potential outcomes at once, purely so we can check our methods against ground truth. Real analysts never get this luxury; we use it here only as a teaching and testing tool.

---

## 2. The Concept

### 2.1 Potential outcome vs. counterfactual: a precise distinction

These two terms are often used loosely as synonyms, but it's worth being precise:

- A **potential outcome**, $Y_i(1)$ or $Y_i(0)$, is defined for a person *before* we know which treatment they'll receive. Both are, in principle, equally "real" possibilities up until treatment assignment happens.
- A **counterfactual** outcome is the potential outcome that did **not** occur — defined *after* the fact, relative to what actually happened. If Alice actually drank coffee ($T_{\text{Alice}}=1$), then $Y_{\text{Alice}}(1)$ is her *factual* outcome (it's what happened), and $Y_{\text{Alice}}(0)$ is her *counterfactual* outcome — the "what would have happened instead" question.

So: every counterfactual is a potential outcome, but only the *unrealized* one, viewed after treatment assignment is already known. This distinction matters when reading causal inference papers, where "counterfactual reasoning" specifically emphasizes the impossible-to-observe, contrary-to-fact nature of the comparison.

### 2.2 The "God's-eye view": a teaching device

Because Module 01's fundamental problem makes real data always incomplete, this module uses a simulation trick throughout the rest of this crash course: since *we* are constructing the population in code, we can choose to generate and keep **both** $Y_i(1)$ and $Y_i(0)$ for every simulated individual, then only reveal $Y_i^{\text{obs}}$ (via $T_i$) to whatever "analysis" we write, exactly as a real study would only ever see $Y_i^{\text{obs}}$. This lets us check any estimator's answer against the true, fully-known ATE — a check that is *only* possible in simulation, never with real-world data (where the true ATE is exactly what you're trying to find out). Every code file in this crash course, starting with Module 01's, uses this pattern.

### 2.3 Reading a full potential-outcomes table

Here is a complete God's-eye table for 6 hypothetical people (in a real study you would never see the grayed-out columns — only $T_i$ and $Y_i^{\text{obs}}$):

| $i$ | $Y_i(0)$ | $Y_i(1)$ | $\tau_i = Y_i(1)-Y_i(0)$ | $T_i$ (actual) | $Y_i^{\text{obs}}$ |
|---|---|---|---|---|---|
| 1 | 50 | 55 | 5 | 0 | 50 |
| 2 | 60 | 68 | 8 | 1 | 68 |
| 3 | 45 | 44 | -1 | 0 | 45 |
| 4 | 70 | 82 | 12 | 1 | 82 |
| 5 | 55 | 60 | 5 | 0 | 55 |
| 6 | 65 | 70 | 5 | 1 | 70 |

A few things worth reading off this table directly:

- **The true ATE** is the average of the $\tau_i$ column: $(5+8-1+12+5+5)/6 = 34/6 \approx 5.67$.
- **Individual effects vary** ($\tau_3 = -1$, meaning treatment would have *hurt* person 3, while $\tau_4=12$ is a large positive effect) — the ATE is a population *average*, and it can mask substantial variation. (Module 05 introduces CATE — the conditional average treatment effect — for exactly this reason.)
- **The observed data alone (columns $T_i$, $Y_i^{\text{obs}}$) cannot recover this table.** A real analyst sees only 6 numbers total (one per person), not 12 — half the table is permanently invisible. Any estimate of the ATE from real data has to be built entirely from the $T_i$/$Y_i^{\text{obs}}$ columns, plus assumptions (Module 06) about how treatment was assigned.

### 2.4 SUTVA: an assumption hiding inside the notation itself

Writing "$Y_i(1)$" as a single, well-defined number for person $i$ already assumes something nontrivial, formalized as the **Stable Unit Treatment Value Assumption (SUTVA)**, which has two parts:

1. **No interference**: person $i$'s potential outcomes depend only on *their own* treatment, not on which treatment anyone else received. (Violated, e.g., by a vaccine study, where whether your neighbor is vaccinated affects your own infection risk — this is called "spillover" or "interference.")
2. **No hidden variation in treatment**: there is only one version of "treatment" and one version of "control" — $Y_i(1)$ means the same specific intervention for everyone who receives it. (Violated, e.g., if "treatment" is "receiving *some* dose of a drug" but the dose actually varies widely across patients — then $Y_i(1)$ isn't really one well-defined outcome.)

SUTVA is easy to overlook precisely because it's baked into the notation before any of the "real" analysis starts — Module 06 revisits SUTVA alongside the crash course's other core assumptions, once we've built up the estimators that all quietly depend on it.

---

## 3. Use It: Code

The full runnable God's-eye simulation for this lesson's table-style reasoning (built to arbitrary population size, not just 6 toy rows) is in `code/potential_outcomes_demo.py`. It:

1. Generates $Y_i(0)$ and $Y_i(1)$ for every simulated individual (the God's-eye view, §2.2).
2. Assigns treatment $T_i$ and computes $Y_i^{\text{obs}}$ (§2.3's rightmost columns).
3. Computes and prints the **true** ATE from the full table, and separately the **naive observed-data** difference in means — deliberately keeping both so you can compare them directly, the same pattern used in Module 01's code.
4. Prints a small preview of the full table (à la §2.3) alongside what a real analyst would actually get to see, so the gap between the two is visible line by line, not just in the summary statistics.

---

## Exercises

1. Using §2.3's table, compute the true ATE two different ways: (a) averaging the $\tau_i$ column directly, and (b) computing $\frac{1}{6}\sum Y_i(1) - \frac{1}{6}\sum Y_i(0)$ separately and subtracting. Confirm both give the same answer, and explain in one sentence why they must always agree (hint: this is really just the linearity of averaging).
2. From §2.3's table, compute the *naive* difference in means using only the observed columns ($T_i$, $Y_i^{\text{obs}}$): average $Y_i^{\text{obs}}$ among the treated ($i=2,4,6$) minus average $Y_i^{\text{obs}}$ among the untreated ($i=1,3,5$). Compare this to the true ATE from Exercise 1 — are they the same or different, and if different, can you spot anything about how $T_i$ was assigned in the table that might explain the gap? (Look at whether people with higher $Y_i(0)$ values seem more or less likely to have $T_i=1$.)
3. Run `code/potential_outcomes_demo.py` twice: once where treatment assignment $T_i$ is generated completely independently of $Y_i(0)$ and $Y_i(1)$ (e.g., a fair coin flip), and once where $T_i$ depends on $Y_i(0)$ (e.g., people with a higher $Y_i(0)$ are more likely to get $T_i=1$, mimicking Exercise 2's pattern). Compare the naive observed-data difference to the true ATE in both runs. In which run does the naive estimate come closer to the truth, and why?
4. For each of SUTVA's two parts (§2.4), describe a realistic scenario, other than the vaccine and drug-dose examples already given, where that part of SUTVA would be violated.

## Key Terms

| Term | What it actually means |
|---|---|
| Counterfactual | The potential outcome that did *not* occur for a given individual, defined relative to what actually happened |
| God's-eye view | A simulation technique used only for teaching/testing: generating both $Y_i(1)$ and $Y_i(0)$ so an estimator's output can be checked against a known true effect |
| SUTVA (Stable Unit Treatment Value Assumption) | The assumption that an individual's potential outcomes depend only on their own treatment (no interference) and that there's only one well-defined version of each treatment level (no hidden variation) |
| Interference / spillover | A violation of SUTVA's "no interference" part, where one unit's treatment affects another unit's outcome |
