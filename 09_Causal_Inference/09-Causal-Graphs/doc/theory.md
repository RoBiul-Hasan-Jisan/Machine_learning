# 09. Causal Graphs

## Learning Objectives

- State the causal Markov assumption precisely, and explain what it licenses you to assume about a DAG
- Distinguish a causal DAG from a mere correlation diagram
- Identify parents, children, ancestors, descendants, and paths in a DAG
- Recognize the three elementary building blocks — chains, forks, colliders — that every DAG is built from (formalized fully in Module 11)

---

## 1. The Problem: Formalizing What a DAG Is Actually Claiming

Module 03 introduced causal DAGs informally, as a picture of "which variable directly causes which." This module makes that picture formally precise: exactly what assumption does drawing an arrow $A \to B$ commit you to, and what does the *absence* of an arrow between two variables commit you to? Getting this right is essential before Modules 10–16 can use DAGs to derive rigorous rules (d-separation, the backdoor criterion) rather than just intuitive diagrams.

---

## 2. The Concept

### 2.1 What a DAG is, formally

A **causal DAG** (Directed Acyclic Graph) $G = (V, E)$ consists of a set of nodes $V$ (variables) and a set of directed edges $E$ (arrows), with two structural requirements:

- **Directed**: every edge $A \to B$ has a direction, representing "$A$ is a direct cause of $B$" (relative to the other variables in the graph — see §2.3).
- **Acyclic**: there is no sequence of edges that starts and ends at the same node, e.g., no $A \to B \to C \to A$. This encodes the assumption that causation doesn't loop back on itself in an instant — a cause always precedes its effect in time, even if the graph doesn't show time explicitly.

### 2.2 Graph vocabulary you'll need going forward

For a node $X$ in a DAG:

| Term | Meaning |
|---|---|
| Parents of $X$ | Nodes with a direct edge into $X$ (its immediate, direct causes) |
| Children of $X$ | Nodes $X$ has a direct edge into |
| Ancestors of $X$ | Every node with a directed path (one or more edges, following arrow direction) into $X$ |
| Descendants of $X$ | Every node reachable from $X$ by following a directed path |
| Path | Any sequence of edges connecting two nodes, **regardless of arrow direction** — this is broader than "directed path," and it's the notion Module 11 needs to talk about non-causal (e.g., confounding) routes between variables |

The distinction between "directed path" and plain "path" matters enormously starting in Module 11: a **causal** path from $T$ to $Y$ follows arrows forward the whole way ($T \to \cdots \to Y$); a **non-causal** path can travel backward against an arrow at some point (e.g., $T \leftarrow Z \to Y$, Module 01's confounding structure) and still connect $T$ and $Y$, creating an association that has nothing to do with $T$ causing $Y$.

### 2.3 The causal Markov assumption: what a DAG actually claims about the data

Drawing a DAG is not just an informal sketch — it's a formal claim, called the **causal Markov assumption** (or causal Markov condition): given its parents, every variable is independent of its non-descendants. Formally, for every variable $X_i$:

$$X_i \perp \{\text{non-descendants of } X_i\} \;\big|\; \text{parents}(X_i)$$

Intuitively: once you know a variable's *direct* causes, learning about anything else that isn't downstream of it (any of its non-descendants) tells you nothing more about it. All of the information other variables carry about $X_i$ is fully "screened off" by its parents. This single assumption is what will let Module 10 turn a DAG into a precise statement about how the joint probability distribution of all the variables must factor, and what will let Module 12 read conditional independencies directly off the graph's structure (d-separation) rather than having to derive them from scratch for every DAG.

### 2.4 What absence of an arrow claims — and what a DAG does not claim

Two clarifications worth being explicit about, since both are easy to get backward:

- **An arrow $A \to B$ does not, by itself, say how strong the effect is**, or even its sign — only that $A$ is one of $B$'s direct causes, given everything else in the diagram. Strength and sign are numerical questions the DAG structure is silent on; Modules 04–05's ATE machinery is what quantifies them.
- **The absence of an arrow between two variables is itself a substantive claim** — specifically, that neither is a *direct* cause of the other (once the other variables in the diagram are accounted for). This is often the most consequential, and most debatable, part of drawing a DAG: leaving an arrow out asserts something, just as drawing one in does. A DAG that's missing a real causal arrow (an unmeasured or overlooked confounder, say) is exactly the failure mode behind unconfoundedness violations from Module 06 — the DAG is implicitly claiming that variable doesn't exist or doesn't matter, when in the real system it does.

### 2.5 Three elementary building blocks, previewed

Module 08 §2.3 already hinted that confounders, mediators, and colliders behave completely differently. In DAG terms, these correspond to the **three elementary junction types** any two variables can be connected through, via one intermediate node $M$:

```
Chain:    T --> M --> Y     (M is a mediator; T's effect on Y flows THROUGH M)
Fork:     T <-- M --> Y     (M is a common cause / confounder of T and Y)
Collider: T --> M <-- Y     (M is a common effect of T and Y)
```

Every DAG, however large, is built entirely out of these three shapes repeated and combined. Module 11 works out, rigorously, exactly when each junction transmits association between $T$ and $Y$ and when conditioning on $M$ blocks or creates that association — the formal version of Module 08 §2.3's warning.

---

## 3. Use It: Code

`code/dag_structure_demo.py` builds a small DAG programmatically (as an adjacency structure), then:

1. Computes parents, children, ancestors, and descendants for each node, so you can check your own by-hand reading of a diagram against code.
2. Enumerates all *paths* (not just directed paths) between two chosen nodes, illustrating §2.2's distinction — some of the enumerated paths will follow arrows the whole way (causal), and some will not (non-causal, potentially confounding).
3. Classifies each three-node subpath running through an intermediate node as a chain, fork, or collider, based purely on edge directions — a mechanical version of the pattern-matching Module 11 will build estimators and blocking rules around.

---

## Exercises

1. Draw (or describe) a DAG with 5 variables of your own choosing, including at least one chain, one fork, and one collider structure somewhere in the diagram. Identify each of the three structures explicitly.
2. For your DAG from Exercise 1, pick one node and list its parents, children, ancestors, and descendants.
3. Explain, in your own words, why the causal Markov assumption (§2.3) is doing real work — i.e., describe a scenario where it could plausibly be *false* for a real system (hint: think about what happens if there's an unmeasured variable that's a direct cause of two variables you've already connected some other way in the diagram).
4. Run `code/dag_structure_demo.py` on the confounder DAG from Module 03 ($Z\to T$, $Z\to Y$, $T\to Y$) and confirm the code correctly identifies the backward-then-forward path $T \leftarrow Z \to Y$ as non-causal, alongside the direct causal path $T \to Y$.

## Key Terms

| Term | What it actually means |
|---|---|
| Causal DAG | A directed, acyclic graph where an edge $A\to B$ asserts "$A$ is a direct cause of $B$," given the other variables shown |
| Causal Markov assumption | The formal claim that every variable is independent of its non-descendants, given its parents |
| Path | Any sequence of edges connecting two nodes, regardless of arrow direction (broader than "directed path") |
| Chain | The junction $T \to M \to Y$; $M$ is a mediator on a causal pathway |
| Fork | The junction $T \leftarrow M \to Y$; $M$ is a common cause (confounder) |
| Collider | The junction $T \to M \leftarrow Y$; $M$ is a common effect |
