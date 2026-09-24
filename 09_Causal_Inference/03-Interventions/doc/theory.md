# 03. Interventions

## Learning Objectives

- Explain the difference between $P(Y \mid X=x)$ (conditioning) and $P(Y \mid do(X=x))$ (intervening)
- Read a simple causal DAG (directed acyclic graph) and identify confounding paths in it
- Explain, using a DAG, why $do(\cdot)$ "cuts" incoming arrows into the intervened variable
- Simulate an intervention directly and compare it to a purely observational (conditioned) analysis of the same system

---

## 1. The Problem: Conditioning Is Not the Same as Intervening

Modules 01–02 built potential outcomes from the individual up. This module introduces a second, complementary language for the same ideas — **structural causal models** and the **$do(\cdot)$ operator** — because it makes one specific confusion impossible to make by accident: the difference between *observing* that $X$ happens to equal $x$, versus *forcing* $X$ to equal $x$.

$P(Y \mid X=x)$ answers: "among the people who happen to have $X=x$, what's the distribution of $Y$?" This is a statement about a subpopulation you filter down to. $P(Y \mid do(X=x))$ answers a different question: "if I took the *entire* population and forced everyone's $X$ to be $x$, what would the distribution of $Y$ become?" This is a statement about an action performed on the whole system. Module 01's coffee example is exactly this confusion: $P(\text{lifespan} \mid \text{coffee}=\text{high})$ (conditioning — just look at people who drink a lot of coffee) was large, while $P(\text{lifespan} \mid do(\text{coffee}=\text{high}))$ (intervening — force everyone to drink a lot of coffee) was, by construction, no different from the baseline.

---

## 2. The Concept

### 2.1 Causal DAGs: drawing the assumptions

A **causal DAG** (directed acyclic graph) is a diagram where an arrow $A \rightarrow B$ means "$A$ is a direct cause of $B$." For Module 01's coffee example:

```
        Z (income / free time)
       /                       \
      v                         v
  T (coffee)  ---------->   Y (lifespan)
   (drawn as having no real effect in that example, but shown here for generality)
```

$Z$ is a **confounder**: it has an arrow into both $T$ and $Y$, creating a "backdoor path" $T \leftarrow Z \rightarrow Y$ that produces correlation between $T$ and $Y$ with no arrow directly connecting them being required at all. This is the graphical picture of exactly what Module 01's numeric example showed.

### 2.2 The $do(\cdot)$ operator: surgery on the graph

Formally, $do(T=t)$ means: take the causal DAG, **delete every incoming arrow into $T$** (in the diagram above, the $Z \to T$ arrow disappears), and set $T=t$ for everyone, regardless of what $Z$ is. This is often called "graph surgery." After the surgery:

```
        Z (income / free time)
                                \
                                 v
  T (coffee) = t, for EVERYONE     Y (lifespan)
   (no longer influenced by Z at all)
```

$Z$ still affects $Y$ directly if such an arrow exists, but $Z$ no longer affects $T$ — the backdoor path $T \leftarrow Z \rightarrow Y$ is severed. This is the graphical reason $P(Y \mid do(T=t))$ differs from $P(Y \mid T=t)$: conditioning ($P(Y\mid T=t)$) leaves the $Z\to T$ arrow intact and just filters to the subpopulation where it happened to produce $T=t$; intervening ($do$) actively removes that arrow and sets $T=t$ for the whole population, confounder and all.

### 2.3 Connecting $do(\cdot)$ back to potential outcomes

$do(\cdot)$ and potential outcomes (Modules 01–02) are two notations for compatible ideas:

$$E[Y \mid do(T=1)] = E[Y_i(1)] \qquad E[Y \mid do(T=0)] = E[Y_i(0)]$$

$$\text{ATE} = E[Y\mid do(T=1)] - E[Y \mid do(T=0)] = E[Y_i(1)] - E[Y_i(0)]$$

In words: "what would the average outcome be if I intervened to set everyone's treatment to 1" is the same quantity as "the average of everyone's $Y_i(1)$ potential outcome" — both are asking about a hypothetical world where treatment was assigned in a specific way, independent of whatever actually determined treatment in the real data. Module 05 works entirely in this language to define and estimate the ATE; keep this equivalence in mind as a translation key between the two notations you'll see across the causal inference literature.

### 2.4 A randomized experiment is $do(\cdot)$, performed for real

Why do randomized controlled trials (RCTs) work? A coin flip assigning treatment is, physically, exactly the graph surgery from §2.2: by construction, the coin's outcome has **no incoming arrow** from $Z$ or anything else about the individual — it's causally disconnected from every pre-existing confounder. So in a true RCT, $P(Y\mid T=t)$ (what you actually observe, conditioning on the realized coin flip) and $P(Y \mid do(T=t))$ (the causal quantity you actually want) **coincide**, because there was never a backdoor path to sever in the first place. This is the deep reason randomization is often called the "gold standard": it makes the easy-to-compute quantity and the hard-to-define quantity the same thing. Module 05 builds directly on this fact to justify the standard RCT estimator.

### 2.5 A population-level question, not a per-person one

One thing $do(\cdot)$, as used in this module, does **not** give you: an answer for one *specific*, already-observed individual. $P(Y\mid do(T=1))$ describes what would happen if the *whole population* were forced into treatment — it says nothing about, say, "Alice specifically took the treatment and we observed her outcome; what would have happened to *her*, specifically, had she not?" That sharper, individual-level question is a genuinely different (and harder) kind of hypothetical intervention, and it's the entire subject of Module 04, which extends the structural causal model introduced in §2.2 into a procedure for answering exactly that.

---

## 3. Use It: Code

`code/intervention_demo.py` builds a small structural causal model (SCM) — a set of equations, each variable computed as a function of its direct causes plus independent noise, exactly matching the DAG in §2.1 — and then:

1. Simulates the **observational** data: $Z$, then $T$ (as a function of $Z$), then $Y$ (as a function of both $T$ and $Z$), and computes $E[Y \mid T=1] - E[Y\mid T=0]$ directly from this data (conditioning).
2. Simulates an explicit **intervention**: regenerates the population, but this time sets $T=1$ for everyone (ignoring $Z$ entirely, per §2.2's graph surgery) to get $E[Y\mid do(T=1))]$, then repeats with $T=0$ for everyone to get $E[Y \mid do(T=0)]$, and subtracts.
3. Prints both numbers side by side, so the gap between "conditioning" and "intervening" — purely a consequence of $Z$ being a confounder — is visible directly, extending Module 01's numeric demonstration into the DAG/$do(\cdot)$ language introduced here.

---

## Exercises

1. Draw (on paper or describing in words) the causal DAG for a scenario of your choosing with at least one confounder, one treatment, and one outcome. Identify the backdoor path(s) explicitly.
2. For the DAG in §2.1, explain in your own words what would change about the diagram, and about whether $P(Y\mid T=t) = P(Y \mid do(T=t))$, if there were *no* arrow from $Z$ to $T$ (i.e., $Z$ still affects $Y$ directly, but has nothing to do with who gets treated).
3. Run `code/intervention_demo.py` and record both the conditioning-based estimate and the two do-operator-based estimates (for $do(T=1)$ and $do(T=0)$). Then modify the simulation so that $Z$ no longer affects $T$ at all (i.e., $T$ is assigned by a fair coin flip, independent of $Z$), matching Exercise 2's scenario, and rerun. Confirm the conditioning-based estimate now matches the do-operator-based estimate closely, illustrating §2.4's point about randomization.
4. Using §2.3's equivalence between $do(\cdot)$ and potential outcomes, rewrite the ATE formula from Module 01 (§2.4 there) purely in $do(\cdot)$ notation, without using any potential-outcome symbols ($Y_i(1)$, $Y_i(0)$) at all.

## Key Terms

| Term | What it actually means |
|---|---|
| Causal DAG | A directed acyclic graph where an arrow $A\to B$ represents "$A$ is a direct cause of $B$," used to represent causal assumptions visually |
| Backdoor path | A non-causal path connecting treatment and outcome through a common cause (confounder), which creates correlation without a direct causal effect |
| $do(\cdot)$ operator | A formal operation representing an intervention: deleting all incoming arrows into a variable and setting it to a fixed value for the whole population |
| Graph surgery | Another name for what $do(\cdot)$ does to a causal DAG: removing incoming edges into the intervened variable |
| Structural causal model (SCM) | A set of equations, one per variable, expressing each variable as a function of its direct causes plus independent noise — a DAG made computationally concrete |
