# 36. Do-calculus

## Learning Objectives

- State the three rules of do-calculus and what each licenses
- Explain that the rules are complete: an effect is identifiable from observational data if and only if do-calculus can derive it
- Derive the **front-door adjustment** with do-calculus
- Verify the front-door formula numerically when the backdoor criterion fails

---

## 1. The Problem

The backdoor criterion (Module 14) only covers one route. What if the confounder $U$ is unmeasured, so no adjustment set exists? Pearl's **do-calculus** is a set of rules for transforming expressions with $do(\cdot)$ into $do$-free ones whenever the graph allows it.

## 2. The Concept

### 2.1 Notation

$G_{\overline X}$: graph with arrows **into** $X$ removed. $G_{\underline X}$: arrows **out of** $X$ removed. $Z(W)$ means $Z$ minus ancestors of $W$ in $G_{\overline X}$.

### 2.2 The three rules

1. **Insertion/deletion of observations:** $P(y\mid do(x),z,w)=P(y\mid do(x),w)$ if $(Y\perp Z\mid X,W)$ in $G_{\overline X}$.
2. **Action/observation exchange:** $P(y\mid do(x),do(z),w)=P(y\mid do(x),z,w)$ if $(Y\perp Z\mid X,W)$ in $G_{\overline X\underline Z}$.
3. **Insertion/deletion of actions:** $P(y\mid do(x),do(z),w)=P(y\mid do(x),w)$ if $(Y\perp Z\mid X,W)$ in $G_{\overline X\overline{Z(W)}}$.

Rule 2 is why the backdoor adjustment works; rule 3 lets you drop an intervention that cannot affect $Y$.

### 2.3 Completeness

Shpitser–Pearl and Huang–Valtorta showed the rules are **complete** for identification: if no sequence of rule applications removes $do(\cdot)$, the effect is not identifiable from $P(V)$ without further assumptions. This is why Module 17's non-identified example cannot be rescued by cleverness.

### 2.4 Front-door criterion

DAG: $T\to M\to Y$, with an unmeasured $U\to T$ and $U\to Y$, and **no** direct $T\to Y$ edge. $M$ intercepts all directed paths from $T$ to $Y$; $T$ and $M$ have no unblocked backdoor path; every backdoor path from $M$ to $Y$ is blocked by $T$. Then

$$P(y\mid do(t))=\sum_m P(m\mid t)\sum_{t'}P(y\mid t',m)\,P(t').$$

Derivation in two steps: (i) $P(m\mid do(t))=P(m\mid t)$ (no confounding of $T\to M$); (ii) $P(y\mid do(m))=\sum_{t'}P(y\mid m,t')P(t')$ (adjust for $T$ to block $M\leftarrow T\leftarrow U\to Y$). Chain them: $P(y\mid do(t))=\sum_m P(m\mid do(t))P(y\mid do(m))$.

### 2.5 Caveat

The front-door assumptions (full mediation, no $U$ effect on $M$) are strong and rarely plausible in practice; the example shows what identification *can* look like when backdoor adjustment is impossible.

## 3. Use It: Code

`code/front_door_demo.py` builds a binary model with an unmeasured $U$, shows the naive contrast is biased and that adjusting for $M$ does not help, then computes the front-door estimate from observed data and compares it with the true effect from simulating $do(T)$.

## Exercises

1. Add a direct edge $T\to Y$. Why does the front-door formula fail?
2. Add $U\to M$. What breaks?
3. Which rule justifies step (ii)?

## Key Terms

| Term | What it actually means |
|---|---|
| Do-calculus | Three graphical rules for rewriting interventional distributions |
| Completeness | If the rules cannot remove $do$, the effect is not identifiable |
| Front-door criterion | Identification via a fully mediating, unconfounded mediator |
