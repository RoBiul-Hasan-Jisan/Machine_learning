# 14. The Backdoor Path Criterion

## Learning Objectives

- State the backdoor criterion precisely, as a two-part test on a candidate adjustment set
- Explain why each of the criterion's two conditions is necessary, using a counterexample for each
- Apply the criterion to identify a valid adjustment set on a moderately complex DAG
- State and apply the backdoor adjustment formula, connecting it back to Module 07's stratification estimator as a special case

---

## 1. The Problem: A General Test for "Is This Adjustment Set Good Enough?"

Module 13 defined confounding precisely as an open backdoor path, and showed unconfoundedness holds exactly when a conditioning set blocks every such path. But given an actual DAG with many variables, how do you *find* a valid conditioning set systematically, rather than by inspection and hoping you didn't miss a path (as Module 11 §2.5's busy example showed can easily go wrong)? The **backdoor criterion**, due to Judea Pearl, is exactly this: a precise, checkable test that a candidate set $\mathbf{Z}$ can be run through to confirm — or rule out — that adjusting for $\mathbf{Z}$ correctly identifies the causal effect of $T$ on $Y$.

---

## 2. The Concept

### 2.1 The backdoor criterion, stated

A set of variables $\mathbf{Z}$ satisfies the **backdoor criterion** relative to an ordered pair $(T, Y)$ if:

1. **No node in $\mathbf{Z}$ is a descendant of $T$**, and
2. **$\mathbf{Z}$ blocks every backdoor path between $T$ and $Y$** (every path starting with an arrow into $T$, in the d-separation sense from Module 12).

If $\mathbf{Z}$ satisfies both conditions, then $\mathbf{Z}$ is a **valid adjustment set**, and the causal effect of $T$ on $Y$ is identified by the **backdoor adjustment formula**:

$$P(y \mid do(t)) = \sum_{\mathbf{z}} P(y \mid t, \mathbf{z})\, P(\mathbf{z})$$

This formula is the fully general version of Module 07's stratified estimator: "stratify by $\mathbf{Z}$, estimate the effect within each stratum, and average, weighted by each stratum's prevalence" — Module 07 derived this for one specific confounder $Z$; the backdoor criterion tells you exactly which sets $\mathbf{Z}$ (potentially several variables, potentially in more complex DAGs) this same computation is valid for.

### 2.2 Why condition 2 is necessary: an unblocked path leaks confounding

This is just Module 13's definition, restated as a checklist item: if some backdoor path is left open by $\mathbf{Z}$, that path contributes non-causal association to $P(y\mid t, \mathbf{z})$, and the adjustment formula's average over $\mathbf{z}$ will be biased for the true causal effect — exactly the failure mode in Module 06's confounded examples, where the relevant confounder was never conditioned on at all.

### 2.3 Why condition 1 is necessary: don't adjust for a descendant of treatment

This is the condition that's easy to overlook, and the reason the criterion needs two parts rather than just "block the backdoor paths." Consider adjusting for a **mediator** $M$ (a descendant of $T$, since $T \to M$): Module 11 §2.1 already showed conditioning on a chain node blocks that pathway — meaning adjusting for $M$ would remove part of $T$'s true effect on $Y$, giving a biased (too small) estimate of the *total* causal effect, even though $M$ isn't on any backdoor path at all. The backdoor criterion's condition 1 rules this out directly, by simply forbidding any descendant of $T$ from being in the adjustment set — sidestepping the mediator problem entirely, rather than requiring you to separately reason about it every time.

A second, subtler failure condition 1 also guards against: adjusting for a descendant of a **collider** that itself descends from $T$ can open a previously-blocked path (Module 11 §2.3's "conditioning on a collider's descendant" extension) in ways that are easy to miss without the explicit rule. Excluding all descendants of $T$ as a blanket rule avoids needing to check every such subtlety by hand.

### 2.4 Worked example: applying the criterion on Module 11's busy DAG

Recall the DAG: $Z\to T$, $Z\to Y$, $T\to M\to Y$, $T\to C\leftarrow Y$. Check three candidate adjustment sets for identifying $T$'s effect on $Y$:

- **$\mathbf{Z}_1 = \{Z\}$**: Condition 1 — is $Z$ a descendant of $T$? No ($Z$ is a parent of $T$, not a descendant). Condition 2 — does $\{Z\}$ block every backdoor path? The only backdoor path is $T\leftarrow Z\to Y$, and conditioning on $Z$ blocks it (Module 11 §2.2). **Both conditions hold — $\{Z\}$ is a valid adjustment set.**
- **$\mathbf{Z}_2 = \{Z, M\}$**: Condition 1 — is $M$ a descendant of $T$? Yes ($T\to M$). **Condition 1 fails — not a valid adjustment set**, regardless of what condition 2 would say. (This matches §2.3's warning: $M$ is a mediator, and including it would remove part of the true effect.)
- **$\mathbf{Z}_3 = \{Z, C\}$**: Condition 1 — is $C$ a descendant of $T$? Yes ($T \to C$). **Condition 1 fails again** — and independently, Module 11 §2.5's table already showed conditioning on $C$ opens the previously-blocked collider path, so condition 2 would fail here too even if condition 1 didn't.

Only $\{Z\}$ passes — matching the intuition built up since Module 01 that $Z$ alone is exactly what needs to be adjusted for here, now derived from a general, mechanical two-part test rather than from case-specific reasoning about this particular diagram.

### 2.5 Multiple valid adjustment sets can exist

The backdoor criterion doesn't necessarily pick out a unique set — in a DAG with several confounders arranged differently, more than one distinct set $\mathbf{Z}$ can each independently satisfy both conditions, and each would give a valid (in expectation, unbiased) estimate of the causal effect. When multiple valid sets exist, practical considerations outside the criterion itself — which variables are actually measured, which adjustment set is smallest (fewer strata, better positivity per Module 06 §2.3), which is measured most reliably — guide the choice among them. Module 16 returns to this point directly, once the disjunctive cause criterion (Module 15) has introduced a second, complementary way of arriving at a valid set.

---

## 3. Use It: Code

`code/backdoor_criterion_demo.py` extends Module 12's `d_separated` function into a full `satisfies_backdoor_criterion(Z_candidate, T, Y, edges)` checker implementing both conditions of §2.1, then:

1. Reproduces §2.4's three candidate-set checks programmatically, confirming $\{Z\}$ passes and $\{Z,M\}$, $\{Z,C\}$ both fail (and reports *which* condition failed for each).
2. Runs a brute-force search over every subset of a chosen candidate variable list, printing all valid adjustment sets found — directly extending Module 12's conditioning-set sweep into a genuine backdoor-criterion search.
3. Applies the backdoor adjustment formula numerically to the busy DAG's simulated data using the one valid set found, and confirms the resulting estimate matches the DAG's true, simulated causal effect of $T$ on $Y$.

---

## Exercises

1. For the DAG $A \to T \to Y$, $A \to Y$ (i.e., $A$ confounds $T$ and $Y$ directly, with no other variables), verify by hand that $\{A\}$ satisfies the backdoor criterion, and that the empty set does not.
2. For the DAG in Exercise 1, extended with an additional mediator $T \to M \to Y$, verify that $\{A, M\}$ fails the backdoor criterion (identify which condition fails), while $\{A\}$ alone still passes.
3. Run `code/backdoor_criterion_demo.py`'s brute-force search on Module 11's busy DAG, using candidate variables $\{Z, M, C\}$. Confirm the only valid adjustment set found is $\{Z\}$ (not the empty set, not any set containing $M$ or $C$).
4. Construct a DAG (on paper or in code) with **two** valid adjustment sets that don't overlap at all (e.g., two different, redundant confounders $Z_1$ and $Z_2$, each independently blocking the same backdoor path in a different way — this requires some care in how you wire the edges; try $T \leftarrow Z_1 \to Z_2 \to Y$ where both $\{Z_1\}$ and $\{Z_2\}$ separately work). Verify both sets pass the criterion independently.

## Key Terms

| Term | What it actually means |
|---|---|
| Backdoor criterion | A two-part test on a candidate set $\mathbf{Z}$: no node in $\mathbf{Z}$ is a descendant of $T$, and $\mathbf{Z}$ blocks every backdoor path between $T$ and $Y$ |
| Valid adjustment set | Any set of variables satisfying the backdoor criterion for a given $(T,Y)$ pair, licensing use of the backdoor adjustment formula |
| Backdoor adjustment formula | $P(y\mid do(t)) = \sum_{\mathbf{z}} P(y\mid t,\mathbf{z})P(\mathbf{z})$; the general form of Module 07's stratified estimator |
