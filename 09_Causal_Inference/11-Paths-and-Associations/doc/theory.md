# 11. Paths and Associations

## Learning Objectives

- Determine, for each of the three elementary junctions (chain, fork, collider), whether association flows through it unconditionally and when conditioning on the middle node changes that
- State the general rule for when a path is "open" (transmits association) versus "blocked"
- Trace, on a multi-node DAG, every path between two variables and determine which are open
- Connect this directly to Module 08's confounder/mediator/collider distinction, now made fully rigorous

---

## 1. The Problem: When Does a Path Create Association?

Module 09 §2.5 introduced three elementary junctions — chain, fork, collider — as the building blocks of any DAG. Module 08 warned that adjusting for the "wrong" variable can either fail to fix bias or actively create it. This module makes that warning precise by answering, junction by junction, exactly one question: **does association flow between the two endpoints of this junction, and how does conditioning on the middle node change the answer?** Once this is settled for all three junction types, Module 12 will assemble the answer into a single general algorithm (d-separation) that works on paths of any length in any DAG.

---

## 2. The Concept

### 2.1 Chain: $T \to M \to Y$

**Unconditional (not conditioning on $M$):** association flows through. If $T$ causes $M$ and $M$ causes $Y$, then $T$ and $Y$ are associated — learning $T$'s value tells you something about $M$, which tells you something about $Y$.

**Conditioning on $M$:** the path is **blocked**. Once you already know $M$'s value, learning $T$'s value gives you no further information about $Y$ — everything $T$ could tell you about $Y$ was *mediated entirely through* $M$, and knowing $M$ directly makes that channel redundant. Formally: $T \perp Y \mid M$ for a pure chain.

This matches Module 08 §2.3's warning about mediators exactly: $M$ sits on the one and only pathway from $T$ to $Y$, so conditioning on it severs that pathway — which is why adjusting for a pure mediator removes part of the true causal effect you were trying to measure, rather than removing bias.

### 2.2 Fork: $T \leftarrow M \to Y$

**Unconditional:** association flows through. A common cause $M$ makes $T$ and $Y$ move together even though neither causes the other — exactly Module 01's confounding structure (there, $M=Z$).

**Conditioning on $M$:** the path is **blocked**. Once $M$'s value is fixed, $T$ and $Y$ no longer share a common source of variation — within any single value of $M$, whatever made $T$ vary is unrelated to whatever made $Y$ vary (assuming no other connecting path exists). Formally: $T \perp Y \mid M$ for a pure fork.

This is precisely why Module 07's stratification estimator works: stratifying on the confounder $Z$ blocks the fork $T \leftarrow Z \to Y$, isolating whatever association remains as attributable to $T$'s direct effect on $Y$ alone.

### 2.3 Collider: $T \to M \leftarrow Y$

**Unconditional:** the path is **already blocked**, with no conditioning needed. $T$ and $Y$ are two separate causes of $M$; absent any other connecting path, they are marginally independent — knowing $T$ tells you nothing about $Y$ if you haven't looked at $M$ at all.

**Conditioning on $M$:** the path becomes **open** — association is *created* where none existed. Once you restrict attention to (or otherwise condition on) a specific value of their shared effect $M$, learning that $T$ was "responsible" for some of $M$'s value changes how much of $M$'s value must be attributable to $Y$, and vice versa — precisely the mechanism in Module 08 §2.4's numeric example. Formally: $T \not\perp Y \mid M$, even though $T \perp Y$ unconditionally, for a pure collider.

**A crucial extension**: conditioning on any *descendant* of a collider has the same unblocking effect, to a lesser degree, as conditioning on the collider itself — because a descendant carries partial information about the collider. If $M \to D$ ("D" a consequence of the collider $M$), conditioning on $D$ partially opens the $T \to M \leftarrow Y$ path even without touching $M$ directly.

### 2.4 The general rule: when is a path blocked?

Putting §2.1–2.3 together, a path between two nodes, given a conditioning set $\mathbf{X}$ (possibly empty), is **blocked** if at least one of the following holds for some node $M$ along the path:

1. $M$ is a chain or fork node on the path ($\to M \to$ or $\leftarrow M \to$), and $M \in \mathbf{X}$ (you're conditioning on it), **or**
2. $M$ is a collider node on the path ($\to M \leftarrow$), and neither $M$ nor any descendant of $M$ is in $\mathbf{X}$ (you're conditioning on neither it nor anything downstream of it).

A path that is **not** blocked is called **open** — it transmits association between its two endpoints. This single rule, applied to *every* path between two variables in a DAG (not just one), is exactly what Module 12 packages into the formal criterion called d-separation.

### 2.5 A four-node worked example

Consider a DAG with $Z \to T$, $Z \to Y$, $T \to M \to Y$, and $T \to C \leftarrow Y$ (a confounder $Z$, a mediator $M$, and a collider $C$, all connecting $T$ and $Y$ simultaneously — a deliberately busy diagram, to practice the general rule on more than one path at once). There are (at least) three paths between $T$ and $Y$:

| Path | Junction type(s) along it | Open with no conditioning? | Open if you condition on $\{Z\}$? | Open if you condition on $\{Z, M\}$? | Open if you condition on $\{Z, C\}$? |
|---|---|---|---|---|---|
| $T \to Y$ (direct) | — (no intermediate node) | Yes (it's the causal effect itself) | Yes | Yes | Yes |
| $T \leftarrow Z \to Y$ | fork at $Z$ | Yes | **No** (blocked — conditioned on fork node) | No | Yes |
| $T \to M \to Y$ | chain at $M$ | Yes | Yes | **No** (blocked — conditioned on chain node) | Yes |
| $T \to C \leftarrow Y$ | collider at $C$ | **No** (blocked — collider, not conditioned on) | No | No | **Yes** (opened — conditioned on collider!) |

Reading the last column: conditioning on $\{Z, C\}$ correctly blocks the confounding path through $Z$, but **accidentally opens** the previously-blocked collider path through $C$ — a new source of bias introduced by "adjusting for" the wrong variable. This table is the fully rigorous version of Module 08's warning, and it's exactly the kind of table Module 14's backdoor criterion is designed to let you construct correctly and automatically, rather than by careful (and error-prone) manual tracing.

---

## 3. Use It: Code

`code/path_blocking_demo.py` extends Module 09's DAG toolkit with a `is_blocked(path, conditioning_set)` function implementing §2.4's rule exactly, then:

1. Verifies each of the three single-junction cases (§2.1–2.3) numerically: simulates data from a chain, a fork, and a collider, and confirms (via a simple conditional-independence check on simulated data) that association appears/disappears exactly where the rule predicts.
2. Reproduces §2.5's full four-path table programmatically for the busy example DAG, printing whether each path is open or blocked under each of the four conditioning sets shown — so you can check the code's answers against the table.

---

## Exercises

1. For a DAG $A \to B \to C \to D$ (a longer chain), determine whether $A$ and $D$ are associated (a) with no conditioning, (b) conditioning on $B$, (c) conditioning on $C$. Explain each answer using §2.1's rule.
2. For a DAG where $T \to Y$ directly, and also $T \to M \leftarrow Y$ (a collider $M$ sitting alongside the direct causal path, not on it), explain why conditioning on $M$ here does **not** remove the true $T\to Y$ effect from your estimate the way conditioning on a mediator would — but does introduce collider bias into it. (Hint: think about what you're actually estimating once collider bias and the true effect are both present in the same conditioned-on comparison.)
3. Run `code/path_blocking_demo.py`'s collider-descendant extension (add a node $D$ with $C \to D$ to the busy DAG from §2.5, and condition on $D$ instead of $C$). Confirm numerically that conditioning on $D$ partially opens the $T \to C \leftarrow Y$ path, even though $D$ isn't the collider itself.
4. Using §2.5's table as a template, construct your own 4-path (or more) DAG connecting two variables of interest via at least one chain, one fork, and one collider, and fill in your own version of the table by hand before checking it against the code.

## Key Terms

| Term | What it actually means |
|---|---|
| Open path | A path along which association can flow between its two endpoints, given a specified conditioning set |
| Blocked path | A path along which association cannot flow, given a specified conditioning set |
| Collider bias (via conditioning) | Association introduced between two variables by conditioning on their common effect (or a descendant of it) |
| Descendant of a collider | Any variable downstream of a collider; conditioning on it partially opens the collider's path, just as conditioning on the collider itself would |
