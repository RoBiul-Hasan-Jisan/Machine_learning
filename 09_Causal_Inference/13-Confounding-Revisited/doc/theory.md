# 13. Confounding Revisited

## Learning Objectives

- Restate confounding fully rigorously, as an open "backdoor path" in the d-separation sense
- Connect this graphical definition back to the informal one from Module 01 and the associational-vs-causal gap from Module 04
- Recognize Simpson's paradox as a vivid symptom of unaddressed confounding, and resolve it using the tools built in Modules 09–12
- State precisely what "no unmeasured confounding" (Module 06's unconfoundedness assumption) means in graph terms

---

## 1. The Problem: Turning "Confounder" Into a Graphical, Checkable Definition

Modules 01 and 08 defined a confounder informally: a common cause of treatment and outcome. Now that Modules 09–12 have built causal DAGs, paths, and d-separation into precise tools, this module gives confounding its final, rigorous definition — one stated entirely in terms of paths and blocking, which is what makes it possible for Module 14 to state a general, checkable criterion (the backdoor criterion) for when adjustment removes confounding bias, rather than relying on case-by-case intuition.

---

## 2. The Concept

### 2.1 Confounding, defined via backdoor paths

A **backdoor path** from treatment $T$ to outcome $Y$ is any path between them that starts with an arrow **pointing into** $T$ (i.e., $T \leftarrow \cdots$). Module 01's fork, $T \leftarrow Z \to Y$, is the simplest possible backdoor path. **Confounding bias exists precisely when at least one backdoor path between $T$ and $Y$ is open** (in the d-separation sense from Module 12) given whatever conditioning set (possibly empty) an analysis is using. This is the fully rigorous version of Module 08 §2.1's two-part informal definition:

- Condition (1) there ("$Z$ is a cause of $T$") is what makes $Z\to T$ (in reverse, $T \leftarrow Z$) exist as an edge at the start of the path.
- Condition (2) there ("$Z$ is a cause of $Y$ through a pathway not through $T$") is what makes the rest of the path reach $Y$ without needing to pass through $T$ again.

### 2.2 Reconnecting to the associational-vs-causal gap

Module 04 §2.3 stated confounding bias as the gap $E[Y\mid T=t] \ne E[Y\mid do(T=t))]$. Here's why an open backdoor path is exactly what produces that gap: $E[Y\mid T=t]$ is a purely associational quantity — it's affected by *every* open path between $T$ and $Y$, causal or not. $E[Y\mid do(T=t)]$, by contrast, is computed after Module 03 §2.2's graph surgery, which severs every arrow *into* $T$ — precisely every backdoor path, by definition (§2.1), since they all start with an arrow into $T$. So $do(\cdot)$ removes exactly the paths that create confounding bias, and leaves untouched exactly the paths (like a direct edge $T\to Y$, or $T \to M \to Y$) that represent the genuine causal effect. The gap between the two quantities is precisely the contribution of the open backdoor paths.

### 2.3 Simpson's paradox as a confounding symptom

**Simpson's paradox** is the striking (but, once you understand backdoor paths, no longer mysterious) phenomenon where a trend appears in several different groups of data but reverses or disappears when the groups are combined. A classic numeric setup: a treatment appears *worse* than no treatment overall, but *better* within every subgroup defined by a covariate $Z$ — because $Z$ is a strong confounder whose distribution differs sharply between the treated and untreated groups.

Simpson's paradox is not a special, separate phenomenon requiring its own theory — it is exactly what an open backdoor path through $Z$ looks like when you compare the marginal (unconditioned, backdoor path open) association to the $Z$-specific (conditioned, backdoor path blocked) associations, and the confounding happens to be strong enough, and shaped in just the right way, to flip the sign rather than merely change the magnitude (as in Module 01's numeric example, where the sign didn't flip, but the magnitude was severely distorted). §3's code constructs a genuine sign-flipping example, so you can see the full paradox, not just a magnitude distortion.

### 2.4 Unconfoundedness, restated in graph terms

Module 06 §2.2 defined unconfoundedness as $(Y_i(0), Y_i(1)) \perp T_i \mid X_i$ — a statement in potential-outcomes notation. In graph terms, this holds precisely when: **conditioning on $X$ blocks every backdoor path between $T$ and $Y$**, i.e., $X$ d-separates $T$ from $Y$ along every backdoor-starting path (though not necessarily along the causal paths themselves, which should remain open — Module 14 makes this distinction precise as part of the backdoor criterion's formal statement). This is the graph-theoretic translation that finally connects the two languages this crash course has used side by side since Module 03: potential outcomes (Modules 01, 02, 04–07) and causal graphs (Modules 03, 09–13) are two notations for the same underlying theory, and unconfoundedness, in particular, is exactly "no open backdoor path" once you have a graph to check it against.

---

## 3. Use It: Code

`code/simpsons_paradox_demo.py`:

1. Constructs a dataset exhibiting a genuine Simpson's paradox: the treatment looks harmful overall but helpful within every level of a confounder $Z$, using deliberately chosen group sizes and effect sizes (a kidney-stone-treatment-style setup, a well-known real example of this exact phenomenon).
2. Computes the marginal (unconditioned) association and the $Z$-specific associations side by side, showing the sign flip directly.
3. Applies Module 07's stratified estimator to recover the correct, sign-consistent effect, and explicitly identifies the open backdoor path ($T\leftarrow Z\to Y$) responsible for the marginal reversal, tying the demonstration back to §2.1's definition.

---

## Exercises

1. In your own words, explain why every fork ($T\leftarrow Z\to Y$) is a backdoor path, but not every backdoor path is a simple fork (hint: think about a longer path like $T \leftarrow Z_1 \to Z_2 \to Y$ — does it still start with an arrow into $T$?).
2. Run `code/simpsons_paradox_demo.py` and confirm the sign of the association flips between the marginal comparison and each stratum-specific comparison. Then compute the correctly stratified estimate and compare its sign to the stratum-specific results.
3. Using §2.4's definition, explain why "no unmeasured confounding" is a strictly graph-dependent claim: the same real-world system could satisfy unconfoundedness given one set of measured covariates $X$ but fail to satisfy it given a smaller set $X' \subset X$. What would have to be different about $X$ versus $X'$ for this to happen?
4. Construct your own small numeric example (extending the Simpson's paradox demo, or from scratch) where a confounder distorts the *magnitude* of an effect without flipping its *sign* (i.e., Module 01's style of distortion, not a full paradox). What's different, structurally, about your example compared to the sign-flipping one in this lesson?

## Key Terms

| Term | What it actually means |
|---|---|
| Backdoor path | A path between treatment and outcome that begins with an arrow pointing into the treatment variable |
| Confounding bias (graphical definition) | The existence of at least one open backdoor path between treatment and outcome, given the conditioning set an analysis uses |
| Simpson's paradox | A trend present within every subgroup of data that reverses when the subgroups are combined, typically caused by a strong confounder |
