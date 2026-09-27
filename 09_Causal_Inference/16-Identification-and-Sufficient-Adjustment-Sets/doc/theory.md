# 16. Identification and Sufficient Adjustment Sets

## Learning Objectives

- Define "identification" precisely, and explain how it differs from estimation
- Synthesize Modules 09–15 into a single practical workflow for identifying a causal effect from observational data
- Explain how the backdoor criterion and the disjunctive cause criterion complement each other in practice
- Recognize the most common real-world identification mistakes, and connect each one to the specific lesson that explains why it's a mistake

---

## 1. The Problem: Bringing It All Together

Every module since 08 has built one piece of machinery: confounding's precise definition (08, 13), the graph language to represent causal assumptions (09, 10), the rules for how association flows through a graph (11, 12), and two complementary criteria for choosing a valid adjustment set (14, 15). This final module of the confounding/DAGs part doesn't introduce a new mechanism — it assembles everything into a single coherent workflow, and names the concept all of it has been in service of: **identification**.

---

## 2. The Concept

### 2.1 Identification vs. estimation: two separate steps

**Identification** asks: *given perfect, infinite data, and a set of causal assumptions (encoded in a DAG), can the causal quantity of interest be expressed as a function of the observed distribution at all?* This is a question about assumptions and mathematics, not about any particular finite dataset. Module 14's backdoor criterion, when satisfied, is exactly a positive answer to an identification question: it certifies that $P(y\mid do(t))$ *can* be written as $\sum_{\mathbf{z}} P(y\mid t,\mathbf{z})P(\mathbf{z})$, a quantity computable in principle from observational data alone.

**Estimation** asks a separate, second question: *given the actual, finite data you have, how do you compute a good numerical estimate of that identified quantity, and how much uncertainty surrounds it?* Module 04's difference-in-means-plus-confidence-interval, and Module 07's stratified estimator, are both estimation tools — but they're only trustworthy *because* an identification argument (Modules 06's assumptions, Module 14's criterion) already established that the quantity they're estimating equals the causal quantity you actually care about. **Identification comes first, logically**: an estimator applied to an unidentified quantity produces a confident-looking number that simply doesn't mean what you want it to mean, however precise it looks (compare this to Module 04 §2.4's confidence interval, or Module 04's own note — a tight CI around a biased number is not progress).

### 2.2 A practical identification workflow, assembled from Modules 09–15

Given a treatment $T$, an outcome $Y$, and a set of measured covariates, here is the workflow this part of the crash course has built, step by step:

1. **Draw the DAG** (Module 09): using domain knowledge, encode every plausible causal relationship among $T$, $Y$, and the measured (and any known-but-unmeasured) covariates.
2. **Identify the roles of each covariate** (Module 08, 11): is it a confounder, a mediator, a collider, an instrument, or unrelated? This determines whether it belongs in an adjustment set at all.
3. **Search for a valid adjustment set** — two complementary routes:
   - If you trust the DAG's structure in detail, apply the **backdoor criterion** (Module 14) directly: exclude descendants of $T$, and confirm the candidate set blocks every backdoor path (Module 12's d-separation).
   - If you're less confident in the exact DAG but confident about which variables cause $T$ or $Y$, apply the **disjunctive cause criterion** (Module 15) as a robust default.
4. **Check the remaining assumptions** (Module 06): positivity/overlap for the chosen adjustment set, and consistency/SUTVA for how treatment and outcome are actually defined and measured.
5. **Only now, estimate**: apply Module 07's stratification (or another estimator, for adjustment sets too large or too continuous for stratification to handle well) to the identified quantity, and quantify uncertainty (Module 04 §2.4's confidence interval logic generalizes here too).

### 2.3 How the two criteria complement each other

Modules 14 and 15 aren't competitors so much as tools suited to different amounts of confidence in the DAG. A natural way to use both together: start with the disjunctive cause criterion (Module 15) as a conservative, DAG-light default that's unlikely to miss a real confounder. If you're willing to commit to a more detailed DAG for a specific analysis, use the backdoor criterion (Module 14) to check whether a *smaller* subset of that generous set is already sufficient — trimming away variables that don't actually sit on any backdoor path, to improve positivity and reduce variance (Module 15 §2.3), while confirming you haven't accidentally dropped something that matters (Module 14 §2.2).

### 2.4 A checklist of common identification mistakes, and where each is explained

| Mistake | What goes wrong | Where it's explained |
|---|---|---|
| Ignoring a real confounder | Naive comparison is biased; an open backdoor path remains | Modules 01, 08 §2.1, 13 §2.1 |
| Adjusting for a mediator | Part of the true causal effect is removed | Module 11 §2.1, Module 14 §2.3 |
| Adjusting for a collider (or its descendant) | Spurious association is created where none existed | Module 08 §2.4, Module 11 §2.3 |
| Adjusting for a pure instrument | No bias reduction, but added estimator variance | Module 15 §2.4 |
| Assuming unconfoundedness without justification | The single most consequential, and least checkable, assumption in the whole framework | Module 06 §2.2 |
| Treating a Markov-compatible DAG as "confirmed" | Multiple DAGs can fit the same data; compatibility isn't proof | Module 10 §2.4 |
| Estimating without first establishing identification | A confident, precise number that doesn't answer the causal question you meant to ask | This module, §2.1 |

This table is worth returning to whenever a real causal analysis feels shaky — most practical mistakes in applied causal inference are a version of one of these seven rows, and each has a specific, named diagnosis rather than being a vague, generic "confounding" worry.

---

## 3. Use It: Code

`code/identification_workflow_demo.py` runs the full five-step workflow from §2.2 end-to-end on a moderately complex, randomly-flavored DAG (built from the toolkit developed across Modules 09, 11, 12, and 14):

1. Defines a DAG with a confounder, a mediator, a collider, and an instrument, all connecting $T$ and $Y$ simultaneously (a busier version of the running example from Modules 11–15).
2. Runs the backdoor criterion checker (Module 14's code) to find every valid minimal adjustment set.
3. Runs the disjunctive-cause-criterion classifier (Module 15's code) to find the robust, DAG-light adjustment set.
4. Estimates the causal effect using both, and using each of the "mistake" sets from §2.4's table, printing all of them side by side — so the entire table of common mistakes, and the two correct approaches, are visible together as one final, consolidated comparison closing out this part of the crash course.

---

## Exercises

1. In your own words, explain why "the confidence interval is narrow" is not, by itself, evidence that an estimate has correctly identified a causal effect. What would you need to check *before* trusting the confidence interval's implications?
2. Pick any real (or realistic, hypothetical) causal question from a field you're familiar with. Sketch the DAG you believe applies, and walk through §2.2's five-step workflow for it explicitly, noting where you're most and least confident in each step.
3. Run `code/identification_workflow_demo.py` and confirm that the backdoor-criterion set and the disjunctive-cause-criterion set produce similar (unbiased) estimates, while each of the "mistake" sets reproduces the specific failure mode described in its row of §2.4's table.
4. Revisit Module 01's very first coffee/lifespan example with the full vocabulary developed since. Write one or two sentences each explaining, precisely, why the naive comparison was biased (§2.4 table, row 1) and what a correct identification argument for that example would need to establish, referencing the specific modules and section numbers that justify each step.

## Key Terms

| Term | What it actually means |
|---|---|
| Identification | Establishing, using causal assumptions (typically a DAG), that a causal quantity can in principle be expressed as a function of the observed data distribution — a question about assumptions, not about any specific finite sample |
| Estimation | Computing a numerical estimate (and its uncertainty) of an already-identified quantity, from actual finite data |
| Identification workflow | The five-step process (draw the DAG, classify covariate roles, find a valid adjustment set, check remaining assumptions, then estimate) assembled from this part of the crash course |
