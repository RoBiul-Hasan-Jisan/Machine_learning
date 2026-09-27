# 12. Conditional Independence and d-Separation

## Learning Objectives

- State the d-separation criterion precisely, as the assembly of Module 11's per-path rule across every path between two variables
- Read conditional independence claims directly off a DAG, without simulating or checking data
- Explain the connection between d-separation and the factorization theorem (Module 10): why blocked-path independencies are guaranteed to hold in any distribution compatible with the DAG
- Distinguish d-separation (a graphical criterion) from statistical independence (a property of a distribution), and explain how the two relate via faithfulness

---

## 1. The Problem: One Rule Per Path Isn't Yet One Rule Per Variable Pair

Module 11 §2.4 gave a rule for whether a *single path* is open or blocked. But two variables in a DAG are typically connected by several paths at once (as in Module 11 §2.5's busy example) — and two variables are associated if **any** path between them is open, not just some particular one you happened to check. This module packages "check every path" into a single named criterion — **d-separation** — that gives a definitive yes/no answer to "are these two variables conditionally independent given this conditioning set, according to the DAG?"

---

## 2. The Concept

### 2.1 d-separation, defined

Two variables $X$ and $Y$ are **d-separated** by a conditioning set $\mathbf{Z}$ if **every** path between $X$ and $Y$ is blocked by $\mathbf{Z}$ (Module 11 §2.4's per-path rule, applied to all paths simultaneously). If $X$ and $Y$ are d-separated by $\mathbf{Z}$, the DAG implies:

$$X \perp Y \mid \mathbf{Z}$$

Conversely, if at least one path between $X$ and $Y$ is left open by $\mathbf{Z}$, then $X$ and $Y$ are **d-connected** given $\mathbf{Z}$, and the DAG does *not* imply independence (though it also doesn't forbid it — d-connection just means the graph doesn't *guarantee* independence; see §2.4).

### 2.2 Worked example: applying d-separation to the busy DAG

Reusing Module 11 §2.5's DAG ($Z\to T$, $Z\to Y$, $T\to M\to Y$, $T\to C \leftarrow Y$), ask: **are $T$ and $Y$ d-separated by $\{Z\}$?** Checking every path (Module 11's table, column "$\{Z\}$"):

- $T\to Y$ (direct): open
- $T\leftarrow Z\to Y$: blocked
- $T\to M\to Y$: open
- $T\to C\leftarrow Y$: blocked

Since at least one path ($T\to Y$ directly, and also $T \to M \to Y$) remains open, $T$ and $Y$ are **d-connected** given $\{Z\}$ — the DAG does *not* claim $T\perp Y\mid Z$. This is exactly right: $T$ genuinely causes $Y$ (the direct edge) and also affects it through the mediator $M$, so of course they should remain associated even after removing the fork through $Z$. d-separation isn't in the business of claiming causal effects don't exist — it's specifically about identifying when *non-causal* (confounding, or collider-induced) association has been fully removed, which is a different question from "are they associated at all."

Now ask instead: **are $T$ and $Y$ d-separated by $\{Z, M\}$?** Every path is now blocked (Module 11's table, column "$\{Z,M\}$" — all four entries are "blocked" except the direct edge itself, which by convention is excluded from this kind of query since it represents the causal effect being investigated, not a path to be blocked). Excluding the direct causal edge, $T \perp Y \mid \{Z, M\}$ **restricted to the non-causal, non-mediating pathways** — meaning any *remaining* association between $T$ and $Y$, once you strip out the direct effect, must be zero, i.e., there's no leftover confounding bias.

### 2.3 Why d-separation guarantees actual statistical independence

This isn't just a diagram-reading convention — it's backed by the factorization theorem from Module 10. If $P$ is Markov compatible with $G$ (i.e., factorizes according to $G$, per Module 10 §2.1), then **every** d-separation implied by $G$ is guaranteed to hold as an actual conditional independence in $P$. This is a real theorem (sometimes stated as: d-separation implies conditional independence, for any distribution compatible with the graph), not a heuristic — it's why Module 11's "trace the paths" exercise, once assembled into d-separation, gives a genuinely reliable answer rather than just a plausible guess. Module 11 §3's code already demonstrated this numerically for individual paths; here, the theorem says it holds simultaneously across *every* path at once, for *any* distribution consistent with the DAG's edges — not just the specific numeric example simulated.

### 2.4 The other direction: faithfulness

d-separation guarantees that *blocked* paths correspond to true independencies (§2.3). But could two variables be *statistically* independent even though the DAG says they're d-connected (some open path exists)? In principle, yes — if, by some coincidence of the specific numeric parameters, two causal effects flowing along different paths happen to exactly cancel out. The assumption that this kind of coincidental cancellation doesn't happen — that every independence you observe in the data corresponds to an actual d-separation in the graph, and not to a lucky numerical accident — is called **faithfulness**. Faithfulness is what would let you go the *other* direction: from observed independencies in data back to constraints on which DAGs are plausible (the starting point of causal discovery algorithms, briefly mentioned in Module 10 §2.4 as out of this crash course's scope). This crash course assumes faithfulness implicitly whenever it uses observed correlations as evidence about a DAG's structure, but is worth flagging explicitly here as a named, separate assumption from the causal Markov assumption (Module 09 §2.3).

---

## 3. Use It: Code

`code/d_separation_demo.py` extends Module 11's path-blocking code into a full `d_separated(X, Y, conditioning_set)` function that enumerates every path between $X$ and $Y$ (using Module 09's path-enumeration code) and checks whether *all* of them are blocked, then:

1. Reproduces §2.2's two queries ($T \perp Y \mid Z$? and $T\perp Y \mid \{Z,M\}$, restricted to non-direct paths) programmatically.
2. Runs the same two queries as actual numerical conditional-independence tests on simulated data (correlation within narrow conditioning slices, as in Module 11's code), confirming the graphical answer and the statistical answer agree — a concrete demonstration of §2.3's theorem.
3. Sweeps over every possible conditioning set for a chosen pair of variables in a slightly larger DAG, printing which sets d-separate them — a practical preview of the search Module 14's backdoor criterion formalizes for the specific purpose of finding *adjustment* sets.

---

## Exercises

1. For the DAG $A \to B \to C \leftarrow D \to E$, determine whether $A$ and $E$ are d-separated by (a) the empty set, (b) $\{B\}$, (c) $\{C\}$, (d) $\{D\}$. Work through each using Module 11's per-path rule before checking your answers against `code/d_separation_demo.py`.
2. Explain in your own words why d-separation is described in §2.2 as being about "non-causal" association specifically — i.e., why finding that two variables are d-connected doesn't contradict the goal of removing confounding bias.
3. Construct a numeric example (in code) where two variables are genuinely d-connected (some open path exists) but happen to be very close to statistically independent due to two effects nearly canceling — illustrating a near-violation of faithfulness (§2.4). (Hint: create two paths with opposite-signed effects of similar magnitude.)
4. Using `code/d_separation_demo.py`'s sweep functionality, find every conditioning set (out of a handful of candidate variables you choose) that d-separates $T$ and $Y$ in Module 11's busy DAG, restricting to non-descendants of $T$ (a restriction Module 14 will explain the reason for). Compare your list to what you'd guess just by inspecting the diagram.

## Key Terms

| Term | What it actually means |
|---|---|
| d-separation | Two variables are d-separated by a set $\mathbf{Z}$ if every path between them is blocked by $\mathbf{Z}$ (Module 11's per-path rule, applied across all paths) |
| d-connection | Two variables are d-connected given $\mathbf{Z}$ if at least one path between them remains open |
| Faithfulness | The assumption that every conditional independence observed in the data corresponds to an actual d-separation in the true DAG, ruling out coincidental cancellation of effects along different paths |
