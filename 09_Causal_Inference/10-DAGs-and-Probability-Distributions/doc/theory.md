# 10. The Relationship Between DAGs and Probability Distributions

## Learning Objectives

- State the factorization theorem connecting a causal DAG to the joint probability distribution of its variables
- Compute a joint distribution's factorization from a DAG, and verify it numerically
- Explain Markov compatibility: what it means for a distribution to be compatible with a graph, and why compatibility is necessary but not sufficient to recover the graph from data alone
- Distinguish the causal content of a DAG (what Module 09 encodes) from the purely statistical content (what this module extracts from it)

---

## 1. The Problem: A DAG Implies Structure on the Data — But What, Exactly?

Module 09's causal Markov assumption said, informally, that a DAG "screens off" each variable from its non-descendants once its parents are known. This module turns that informal statement into an exact, computable formula: given a DAG, exactly how must the joint probability distribution over all its variables be structured? This connects the purely causal object (the DAG, a claim about the world's mechanisms) to a purely statistical object (a joint distribution, describable from data alone) — and getting this connection precise is what makes it possible, starting in Module 12, to read statistical facts (conditional independencies) directly off a causal diagram.

---

## 2. The Concept

### 2.1 The factorization theorem

If a joint distribution $P$ over variables $V = \{X_1, \ldots, X_n\}$ satisfies the causal Markov assumption relative to a DAG $G$ (Module 09 §2.3), then $P$ **factorizes according to $G$**:

$$P(x_1, \ldots, x_n) = \prod_{i=1}^{n} P\big(x_i \mid \text{parents}_G(x_i)\big)$$

This says the full joint distribution — which could otherwise require specifying an enormous table of probabilities for every combination of every variable — collapses into a product of much smaller pieces, one per variable, each conditioned only on that variable's *direct* causes. This is an enormous simplification whenever a DAG has many variables but each has only a few parents, and it's the computational backbone of nearly every algorithm that uses a DAG (from analytic derivations in this crash course to full-scale probabilistic graphical models elsewhere in this curriculum).

### 2.2 Worked example: factorizing the confounder DAG

For Module 03's DAG ($Z\to T$, $Z \to Y$, $T\to Y$), each variable's parents are: $\text{parents}(Z) = \emptyset$, $\text{parents}(T) = \{Z\}$, $\text{parents}(Y) = \{T, Z\}$. The factorization theorem gives:

$$P(z, t, y) = P(z) \cdot P(t \mid z) \cdot P(y \mid t, z)$$

Compare this to the fully general (no independence assumptions at all) chain-rule factorization of any joint distribution, which is always true regardless of any DAG:

$$P(z,t,y) = P(z) \cdot P(t\mid z) \cdot P(y \mid t, z) \qquad \text{(chain rule, always valid, any order)}$$

For *this particular* variable ordering ($Z$, then $T$, then $Y$), the DAG's factorization and the generic chain-rule factorization happen to coincide exactly — because $Y$'s parents in the DAG are $\{T, Z\}$, i.e., *everything* that came before it in this ordering. The DAG's structural claim would show up as a genuine simplification only if some variable's parent set were smaller than "everything before it" in some ordering — for instance, if $Y$ additionally depended on some earlier variable $W$ (with $W$ listed before $Y$ in the chain-rule expansion) but $W$ were *not* a parent of $Y$ in the DAG, the chain rule would still write $P(y\mid t,z,w)$ while the DAG's factorization would correctly simplify this to $P(y\mid t,z)$, dropping the (assumed) irrelevant $w$. §2.4 works through exactly this case.

### 2.3 A DAG with a genuine simplification

Extend the running example with a fourth variable $W$ — say, "weather" — that affects nothing in this system and is affected by nothing in it (causally isolated, but still correlated with the others only insofar as the DAG allows, which here is not at all). The DAG is: $Z \to T$, $Z\to Y$, $T \to Y$, $W$ (isolated, no edges). Listing variables in the order $Z, T, W, Y$, the chain rule always gives:

$$P(z,t,w,y) = P(z)\, P(t\mid z)\, P(w \mid z, t) \, P(y \mid z, t, w)$$

But the DAG's factorization, using each variable's actual parent set, gives:

$$P(z,t,w,y) = P(z)\, P(t\mid z)\, P(w) \, P(y\mid z, t)$$

Two genuine simplifications appear: $P(w\mid z,t)$ collapses to $P(w)$ (since $W$ has no parents at all — it's independent of everything), and $P(y\mid z,t,w)$ collapses to $P(y \mid z, t)$ (since $W$ isn't one of $Y$'s parents). Both of these are testable, falsifiable claims about the data — Module 12 formalizes exactly how to read such claims (conditional independencies) directly off a DAG's structure, without redoing this kind of derivation by hand every time.

### 2.4 Markov compatibility: what it means, and its limits

A distribution $P$ is said to be **Markov compatible** with a DAG $G$ if $P$ factorizes according to $G$ as in §2.1. This is a two-way relationship worth being precise about:

- If the causal Markov assumption holds for the *true* causal DAG generating the data, the resulting distribution *is* compatible with that DAG (§2.1's theorem, in the forward direction).
- But compatibility, checked purely from data, does **not** uniquely identify a single DAG. Multiple different DAGs can be Markov compatible with the exact same joint distribution (these are called **Markov equivalent** graphs) — for instance, $A \to B$ and $A \leftarrow B$ alone are both compatible with any joint distribution over just $\{A, B\}$ that has *some* nonzero association, since neither implies any conditional independence to check against data. Compatibility is a **necessary** condition for a DAG to be correct, but it is not **sufficient** — data alone, without further assumptions (like knowing which variable came first in time, or a randomized intervention breaking the symmetry), generally cannot tell you the direction of a causal arrow, only whether some proposed DAG is *consistent* with the observed statistical pattern.

This is a humbling, important limitation to sit with: everything from Module 09 onward assumes you already have a plausible causal DAG in hand (from domain knowledge, timing information, or experimental design) — this module and the ones following it are about what you can rigorously *do* with a DAG once you have one, not about *discovering* the DAG from data alone (a genuinely harder problem, called causal discovery, outside this crash course's scope).

---

## 3. Use It: Code

`code/factorization_demo.py`:

1. Builds a small discrete joint distribution by explicitly simulating from the factorized form $P(z)P(t\mid z)P(y\mid t,z)$ for the confounder DAG, then empirically estimates the full joint distribution $P(z,t,y)$ from a large number of samples and confirms it matches $P(z)\cdot P(t\mid z) \cdot P(y \mid t,z)$ computed from the same samples — verifying §2.1's factorization numerically rather than just asserting it.
2. Extends to the four-variable example from §2.3 (adding independent $W$), and confirms empirically that $P(w\mid z,t) \approx P(w)$ and $P(y \mid z,t,w) \approx P(y\mid z,t)$ — the two genuine simplifications predicted by the DAG's structure.
3. Constructs two different, Markov-equivalent 2-variable DAGs ($A\to B$ vs. $A \leftarrow B$) and confirms both are compatible with the same simulated joint distribution, illustrating §2.4's non-identifiability point directly.

---

## Exercises

1. For a DAG with variables $A \to B \to C$ (a simple chain), write out the DAG's factorization of $P(a,b,c)$, and compare it to the generic chain-rule factorization in the order $A, B, C$. Are they the same, or does the DAG's version simplify something?
2. For a DAG with variables $A \to C \leftarrow B$ (a collider, with $A$ and $B$ having no edge between them), write out the DAG's factorization of $P(a,b,c)$. What does the factorization imply about the *marginal* relationship between $A$ and $B$ (i.e., $P(a,b)$, summing/integrating out $C$)? (This foreshadows Module 11's treatment of colliders.)
3. Run `code/factorization_demo.py`'s section on the four-variable example, and confirm numerically that $P(w \mid z, t)$ is close to $P(w)$ for several different values of $z$ and $t$ — i.e., that conditioning on $Z$ and $T$ doesn't change your belief about $W$ at all, as the DAG predicts.
4. In your own words, explain why finding that a proposed DAG is "Markov compatible" with your data is reassuring but not conclusive. What kind of additional information (beyond the joint distribution of the observed variables) could help distinguish between Markov-equivalent DAGs in a real problem?

## Key Terms

| Term | What it actually means |
|---|---|
| Factorization theorem | The identity $P(x_1,\ldots,x_n) = \prod_i P(x_i \mid \text{parents}(x_i))$, which holds whenever $P$ satisfies the causal Markov assumption relative to a DAG |
| Markov compatibility | A distribution $P$ is Markov compatible with a DAG $G$ if $P$ factorizes according to $G$ |
| Markov equivalence | Two different DAGs that imply exactly the same set of conditional independencies, and are therefore indistinguishable from observational data alone |
| Causal discovery | The (harder, out-of-scope-here) problem of inferring a causal DAG's structure from data, rather than assuming one is already known |
