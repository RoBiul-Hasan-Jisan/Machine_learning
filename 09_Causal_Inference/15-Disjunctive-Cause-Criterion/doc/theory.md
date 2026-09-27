# 15. The Disjunctive Cause Criterion

## Learning Objectives

- State the disjunctive cause criterion and explain what problem it solves that the backdoor criterion (Module 14) does not
- Explain why "control for every cause of treatment or outcome" is (almost) sufficient without needing the full DAG
- Identify the one category of variable the criterion explicitly excludes, and explain why including it can hurt
- Compare the two criteria's practical trade-offs, and recognize when each is the more useful tool

---

## 1. The Problem: The Backdoor Criterion Requires the Whole DAG

Module 14's backdoor criterion is precise and general — but using it requires knowing the *entire* causal DAG, including exactly which arrows do and don't exist among every pair of measured (and unmeasured) variables, in order to trace every backdoor path correctly. In practice, analysts are rarely handed a fully specified DAG; more often they have a large set of measured covariates and genuine uncertainty about the exact causal structure connecting all of them. VanderWeele's **disjunctive cause criterion** (2019) offers a practical alternative: a simple rule for choosing an adjustment set that provably works *without* needing to know the full DAG, provided you're willing to accept one specific, checkable exclusion.

---

## 2. The Concept

### 2.1 The criterion, stated

**Control for every measured pre-treatment covariate that is a cause of the treatment, a cause of the outcome, or a cause of both — excluding any variable known to be solely an instrumental variable** (a cause of treatment only, with no other pathway to the outcome, covered further in §2.4).

"Disjunctive" refers to the "or": you include a covariate if it's a cause of $T$ **or** a cause of $Y$ (or both) — you don't need to determine *which* category each variable falls into with total precision, only whether it plausibly belongs to the union of the two.

### 2.2 Why this works without knowing the full DAG

Here's the key insight: **every confounder, by definition (Module 08 §2.1), is a cause of both $T$ and $Y$** — so every genuine confounder is automatically included by "cause of $T$ or cause of $Y$." The disjunctive cause criterion doesn't require you to correctly identify which specific variable is *the* confounder, or trace exact backdoor paths — it just requires you to cast as wide a net as "any plausible cause of either $T$ or $Y$," which is guaranteed to catch every confounder without needing to know how they're specifically wired together. This is a real practical advantage: in an applied setting, "is this variable a cause of the outcome?" is often a question domain experts can answer with reasonable confidence, while "does this variable sit on a backdoor path, given the full causal structure of fifteen other variables?" usually is not.

### 2.3 The cost: adjusting for more than you strictly need

Following this criterion typically means adjusting for a **larger** set than the minimal valid set the backdoor criterion would identify (Module 14 §2.5 already noted multiple valid sets can exist; this criterion tends to land on a generous, superset-style choice among them). This cost isn't uniform across every extra variable, though, and it's worth being precise about where it actually comes from:

- A covariate that's purely a cause of $Y$ (unrelated to $T$) — a "precision variable" — is usually **harmless to include, and can even reduce variance**, since it soaks up some of $Y$'s unexplained variation without touching the treatment/outcome relationship at all. §3's code demonstrates this directly.
- The real structural cost comes from **dimensionality**: as more covariates are added to the adjustment set, the number of distinct strata grows, and Module 06 §2.3's positivity assumption gets harder to satisfy (more strata, each with fewer observations — directly connecting to Module 07 §3.1's curse of dimensionality). This cost scales with how many covariates you adjust for and how finely they cut up the data, not with any single variable's role.
- A separate and distinct cost attaches specifically to **instrumental variables** — covered next in §2.4 — which is why the criterion carves them out explicitly rather than leaving "more adjustment is always safe" as the takeaway.

### 2.4 The one exclusion: instrumental variables

An **instrumental variable** is a variable that causes $T$ but has **no effect on $Y$ except through $T$** — i.e., it's a cause of treatment only, sitting on no other pathway to the outcome at all. (A classic example: a randomized encouragement to enroll in a program, which affects who enrolls but has no other route to the outcome except via enrollment itself.) Such a variable is deliberately excluded from the disjunctive cause criterion's adjustment set, and this exclusion matters for a subtle statistical reason: including a variable that predicts $T$ strongly but has no *independent* relationship with $Y$ can, in finite samples, increase the *variance* of the resulting estimate without reducing any bias (since it isn't a confounder to begin with) — a real cost with no corresponding benefit. This doesn't make instrumental variables useless — they're the basis of an entirely different identification strategy (instrumental variable estimation, beyond this crash course's scope) — it just means they don't belong in *this* particular adjustment set.

### 2.5 Comparing the two criteria

| | Backdoor criterion (Module 14) | Disjunctive cause criterion |
|---|---|---|
| Requires the full DAG? | Yes — must trace every backdoor path exactly | No — only requires judging cause-of-$T$-or-$Y$ per variable |
| Typical adjustment set size | Minimal (or one of several minimal options) | Often larger (a generous superset) |
| Risk of getting it wrong | High, if the assumed DAG is wrong or incomplete | Lower — robust to DAG misspecification, given honest cause judgments |
| Main statistical cost | None beyond what's structurally necessary | Reduced positivity margin, higher variance, from over-adjustment |
| What must NOT be included | Any descendant of $T$ (mediators, and their consequences) | The same, plus purely instrumental variables specifically |

Neither criterion is strictly better in every situation — Module 16 discusses how the two are typically used together in practice: the disjunctive cause criterion as a practical default when the full DAG is genuinely uncertain, and the backdoor criterion as the tool for verifying (and potentially trimming down to a smaller, more efficient set) once a specific DAG has been proposed and is being interrogated carefully.

---

## 3. Use It: Code

`code/disjunctive_cause_demo.py`:

1. Builds a DAG with several covariates playing different roles: a genuine confounder, a mediator, a pure instrumental variable, and a covariate that's a cause of $Y$ only (no relation to $T$) — a realistic mix of the categories discussed across Modules 08–15.
2. Constructs the disjunctive cause criterion's adjustment set programmatically (every covariate that's a cause of $T$ or $Y$, excluding the pure instrument) and compares it against the backdoor criterion's minimal valid set from Module 14's checker.
3. Estimates the causal effect using both adjustment sets and compares their bias and estimated variance (across repeated simulated samples), illustrating §2.3's efficiency trade-off directly, and separately demonstrates the variance cost of *including* the excluded instrumental variable, illustrating §2.4's warning.

---

## Exercises

1. For a covariate that's a cause of $Y$ but has no relationship to $T$ at all, explain why the disjunctive cause criterion still includes it (via the "or"), even though Module 08 §2.1 says such a variable isn't a confounder and adjusting for it isn't *necessary* to remove bias. Is including it harmful, according to this lesson?
2. In your own words, explain why an instrumental variable is excluded from the disjunctive cause criterion even though it's a genuine cause of $T$ — what makes it different from a confounder, which is also (in part) a cause of $T$?
3. Run `code/disjunctive_cause_demo.py` and compare the estimated variance of the disjunctive-cause-criterion estimate to the minimal backdoor-criterion estimate, across repeated samples. Which has lower variance, and does the gap match what §2.3 predicts?
4. Run the instrumental-variable-inclusion section specifically, and compare the variance of an estimate that (incorrectly) adjusts for the pure instrument to one that doesn't. Does bias change? Does variance?

## Key Terms

| Term | What it actually means |
|---|---|
| Disjunctive cause criterion | Control for every measured covariate that causes treatment, outcome, or both, excluding pure instrumental variables — a DAG-free alternative to the backdoor criterion |
| Instrumental variable | A variable that affects treatment but has no effect on the outcome except through treatment; deliberately excluded from the disjunctive cause criterion's adjustment set |
| Over-adjustment (efficiency cost) | Including more covariates than strictly necessary in an adjustment set, which can hurt positivity and increase estimator variance without reducing bias |
