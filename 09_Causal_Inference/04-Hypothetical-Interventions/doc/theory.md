# 04. Hypothetical Interventions

## Learning Objectives

- Distinguish a **hypothetical, individual-level** intervention (a counterfactual query about one specific, already-observed unit) from the **population-level** intervention introduced in Module 03
- State and apply the abduction–action–prediction procedure for answering a counterfactual query from a fully specified structural causal model
- Compute a worked counterfactual example by hand for one specific individual, and verify it against code
- Explain why this ability comes at the cost of assuming a fully specified SCM, and connect this back to Module 01's fundamental problem of causal inference

---

## 1. The Problem: A Sharper "What If" Than $do(\cdot)$ Can Answer

Module 03 answered questions of the form: "if we forced **everyone** to take treatment $t$, what would the average outcome be?" — $E[Y \mid do(T=t)]$. This is already a hypothetical, but it's a hypothetical about a *policy applied to a whole population*; it says nothing about any one specific person.

A sharper, more demanding question: *"Alice actually took the treatment, and we observed her outcome. Had she, specifically — with everything else about her held fixed exactly as it was — not taken it, what would have happened to her?"* This is a hypothetical intervention posed at the **individual level**: a genuine counterfactual query about one already-observed unit, not a statement about a population average.

Module 01's fundamental problem of causal inference says this is impossible in general: we never observe both $Y_i(1)$ and $Y_i(0)$ for the same person. This module shows that structural causal models (introduced informally in Module 03 §2.4) give us a way to compute an answer to this sharper question anyway — but only by trading one difficulty for another, as §2.5 makes explicit. Understanding this trade-off is worth the effort: this exact machinery underlies a great deal of modern individual-level causal reasoning — "would this specific patient have recovered without the drug," "would this specific loan applicant have been approved with a higher income" — and any of it that sounds retrospective and person-specific is a hypothetical-intervention (counterfactual) query in this module's sense, not a population-level $do(\cdot)$ query in Module 03's sense.

---

## 2. The Concept

### 2.1 Where this sits in the causal hierarchy

Judea Pearl's "ladder of causation" organizes causal questions into three rungs, and it's useful to see all three side by side, since each of the first two already has its own home earlier in this crash course:

| Rung | Type of question | Notation | Where it's covered |
|---|---|---|---|
| 1. Association | "What do I expect, given what I observe?" | $P(Y\mid X)$ | Ordinary statistics/ML; Module 01 §1 contrasts this with causal questions |
| 2. Intervention | "What would happen if I *acted*, across the population?" | $P(Y \mid do(X))$ | Module 03 |
| 3. Counterfactuals | "What would have happened to *this specific, already-observed unit*, had things been different?" | $P(Y_{T=t} \mid T=t', Y=y')$ | This module |

Each rung requires strictly more than the one below it: Association can be read straight off observational data; Intervention (Module 03) requires a causal model (a DAG or SCM) but no information about any specific individual beyond their treatment; Counterfactuals require both a causal model *and* using what was actually observed about one particular unit to pin down something private to that unit, before asking "what if."

### 2.2 What has to be "private to that unit": exogenous noise

Recall from Module 03 §2.1's DAG that we can write each variable as an equation of its direct causes plus noise (a structural causal model). Written out fully, with the noise terms explicit:

$$Z = U_Z \qquad T = f_T(Z, U_T) \qquad Y = f_Y(T, Z, U_Y)$$

Each $U$ is an **exogenous noise term** — the part of that variable's value not explained by its measured causes, and (crucially for this module) specific to one individual. Two different people with the exact same $Z$ and $T$ can still end up with different $Y$, purely because they have different values of $U_Y$ (unmeasured individual idiosyncrasies — genetics, luck, anything not captured by $Z$ and $T$). Ordinarily $U_Y$ is unobserved and stays that way. The key move in this module is realizing that **if you already know $Z$, $T$, and $Y$ for one specific person, and you're willing to assume you know the exact form of $f_Y$, you can solve the equation backward to recover that person's own $U_Y$** — and once you have it, you can hold it fixed while changing $T$, to ask what $Y$ would have been for *that same person*, noise and all, under a different treatment.

### 2.3 The three-step procedure: abduction, action, prediction

Given a fully specified SCM and full data $(Z=z_0, T=t_0, Y=y_0)$ for one specific unit, computing that unit's counterfactual outcome under a hypothetical treatment $t_1 \ne t_0$ proceeds in three steps:

1. **Abduction**: using the unit's observed values, solve the structural equations for that unit's exogenous noise term(s) — in our running example, solve $y_0 = f_Y(t_0, z_0, U_Y)$ for $U_Y$.
2. **Action**: perform the graph surgery from Module 03 §2.2 — replace $T$'s equation with the constant $t_1$ — while leaving every *other* equation, and the noise value(s) just recovered in step 1, unchanged.
3. **Prediction**: propagate forward through this modified model, using the *same* recovered noise, to compute the counterfactual value of $Y$.

The result, $Y_{T=t_1}(\text{this specific unit})$, is the individual counterfactual outcome — the precise thing Module 01 §2.2 said was fundamentally unobservable. It hasn't stopped being unobservable in reality; what's changed is that we've built a fully specified toy world (an SCM with known equations) in which it becomes computable, and §2.5 is explicit about what that costs.

### 2.4 A complete worked example by hand

Continue the coffee/income/lifespan SCM from Module 03's code, with the same structural equation used there: $Y = 2T + 3Z + U_Y$ (a true causal effect of coffee equal to 2, matching Module 03's `TRUE_EFFECT = 2.0`), where $Z$ is unaffected by $T$ (per the DAG, $Z \to T$ and $Z \to Y$, but no arrow $T \to Z$).

Suppose we observe **Alice**: $Z=1$ (high income), $T=1$ (she drank a lot of coffee), $Y=9$.

**Step 1 — Abduction.** Solve $9 = 2(1) + 3(1) + U_Y$ for Alice's own noise term:

$$U_Y = 9 - 2(1) - 3(1) = 4$$

**Step 2 — Action.** Perform $do(T=0)$: replace Alice's treatment with 0, hypothetically. Her $Z$ is untouched (it isn't caused by $T$), and her personal $U_Y=4$ is carried forward unchanged (it's a property of *her*, not of her treatment).

**Step 3 — Prediction.** Propagate forward with $T=0$, $Z=1$, $U_Y=4$:

$$Y_{T=0}(\text{Alice}) = 2(0) + 3(1) + 4 = 7$$

**Conclusion:** had Alice not been a heavy coffee drinker, her outcome would have been 7 instead of the observed 9. Her *individual* treatment effect, computed exactly (under this assumed SCM), is $9 - 7 = 2$ — matching the model's true per-unit causal effect of coffee exactly, since this particular structural equation happens to give every individual the same treatment effect (no effect heterogeneity here, unlike Module 05's CATE discussion). This exact match is a good sanity check on the procedure, and Exercise 2 asks you to redo this with an SCM that *does* have effect heterogeneity, where the recovered individual effect will differ from person to person.

### 2.5 The cost: you must assume you know the true structural equations

Step 1 of §2.3 only works because we assumed we knew the *exact* functional form $Y = 2T+3Z+U_Y$. In any real application, the true structural equations are never known with certainty — at best, they're estimated from data (often as a linear model, but the true relationship could be nonlinear, or could involve additional causes that weren't measured at all). If the assumed equation is wrong, the "individual counterfactual" computed in step 3 is simply wrong too — and there is generally no way to check this against ground truth, because (Module 01's fundamental problem, still very much in force) the true counterfactual for that individual is never observed either.

In other words: this module doesn't repeal Module 01's fundamental problem. It trades "the individual counterfactual is unobservable" for "the individual counterfactual is computable, but only as reliably as your assumed model of the entire causal system." This is exactly why counterfactual queries are described as the hardest rung of Pearl's ladder (§2.1): they require not just knowing that a causal relationship exists (rung 2's territory), but committing to a fully specified generative story precise enough to solve for someone's private, unobserved noise term — a much stronger, much less checkable assumption than anything used in Modules 01–03.

---

## 3. Use It: Code

`code/hypothetical_intervention_demo.py`:

1. Builds the same coffee/$Z$/$T$/$Y$ structural causal model as Module 03's code (with a known `TRUE_EFFECT`, since we're the ones simulating it), and draws one individual's full data — including their "private" noise term, which is kept hidden from the abduction step exactly as it would be in a real analysis.
2. Runs **abduction** to recover that individual's noise term from their observed $(Z, T, Y)$ alone (without peeking at the hidden noise value used to generate the data).
3. Runs **action + prediction** to compute the individual's counterfactual outcome under the opposite treatment, and compares it to the actual (hidden, but known to us as simulators) true counterfactual for that same unit — confirming the abduction-action-prediction procedure exactly recovers the true individual counterfactual when the assumed SCM is correct.
4. Repeats the whole exercise with a **deliberately wrong** assumed structural equation (e.g., assuming the coefficient on $T$ is 5 instead of the true 2), to make §2.5's warning concrete: the computed counterfactual becomes wrong too, silently, with nothing in the abduction step itself signaling that anything went wrong.

---

## Exercises

**Working the procedure by hand**

1. Redo §2.4's worked example for a second individual, **Bob**, with $Z=0$ (low income), $T=0$ (he did not drink much coffee), and observed $Y=6$. Compute his counterfactual outcome under $do(T=1)$ using the same three-step procedure, and state his individual treatment effect.
2. Modify the structural equation to include effect heterogeneity: $Y = (2 + Z)\,T + 3Z + U_Y$ (so the effect of coffee is larger for high-income individuals). Recompute Alice's counterfactual (§2.4's data: $Z=1, T=1, Y=9$) under this new equation, and note that her recovered $U_Y$ and her individual effect will both differ from §2.4's answer. Why does changing the *assumed* equation change conclusions about a person's data that hasn't itself changed?

**Code and the cost of a wrong model**

3. Run `code/hypothetical_intervention_demo.py` and confirm the abduction-action-prediction procedure recovers the true individual counterfactual when the assumed SCM matches the true one. Then run the "deliberately wrong" section, and record how far off the computed counterfactual is from the true one. Does the size of the error depend on how wrong the assumed coefficient is (try a few different wrong values)?
4. In your own words, explain why "the model gave us a specific, confident-looking number for Alice's counterfactual" is not, by itself, evidence that the number is correct. What kind of evidence (if any) *could* increase your confidence in an individual counterfactual estimate from a real (non-simulated) dataset?

## Key Terms

| Term | What it actually means |
|---|---|
| Hypothetical intervention (counterfactual query) | A question about what would have happened to one specific, already-observed individual under a different treatment, holding everything private to that individual fixed |
| Pearl's causal hierarchy | The three-rung classification of causal questions: association ($P(Y\mid X)$), intervention ($P(Y\mid do(X))$, Module 03), and counterfactuals (this module) |
| Exogenous noise term ($U$) | The part of a variable's value in a structural causal model not explained by its measured causes; treated as private to each individual unit |
| Abduction | Using an individual's fully observed data to solve the structural equations for that individual's exogenous noise term(s) |
| Action | Performing graph surgery (Module 03 §2.2) on the structural model — fixing the treatment variable to a hypothetical value — while keeping the abduced noise unchanged |
| Prediction | Propagating forward through the modified structural model, using the abduced noise, to compute the counterfactual outcome |
