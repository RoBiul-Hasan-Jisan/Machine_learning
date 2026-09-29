# 37. Counterfactual Identification

## Learning Objectives

- Explain why counterfactual (rung 3) quantities are harder to identify than interventional (rung 2) ones
- Define response types (principal strata) and the probabilities of necessity and sufficiency (PN, PS, PNS)
- Compute the Tian–Pearl bounds on PNS from experimental and observational data
- State which assumption (monotonicity) collapses the bounds to a point

---

## 1. The Problem

Module 04 computed individual counterfactuals in a fully specified structural model. Usually we do not know the model. The question becomes: what can data alone say about counterfactual quantities such as "would this patient have survived without the drug?"

## 2. The Concept

### 2.1 Response types

For binary $T$ and $Y$, each unit has a **response type** determined by $(Y(0),Y(1))$:

| Type | $Y(0)$ | $Y(1)$ |
|---|---|---|
| always-recover | 1 | 1 |
| never-recover | 0 | 0 |
| helped | 0 | 1 |
| harmed | 1 | 0 |

An experiment identifies $P(Y(1)=1)$ and $P(Y(0)=1)$, but **not** how units split into types: the same margins are consistent with many joint distributions of $(Y(0),Y(1))$ (a version of the fundamental problem, Module 01).

### 2.2 Probabilities of causation

- **PNS** (necessity and sufficiency): $P(Y(1)=1,\,Y(0)=0)$, the fraction of "helped" units.
- **PN** (necessity): $P(Y(0)=0\mid T{=}1,Y{=}1)$, "given treated and recovered, would they have failed without treatment?"
- **PS** (sufficiency): $P(Y(1)=1\mid T{=}0,Y{=}0)$.

### 2.3 Tian–Pearl bounds

With **experimental** data only, writing $p_1=P(Y(1){=}1)$, $p_0=P(Y(0){=}1)$:
$$\max\{0,\;p_1-p_0\}\le PNS\le\min\{p_1,\;1-p_0\}.$$
With **observational** data as well (joint $P(T,Y)$ and $P(Y)$):
$$\max\{0,\;p_1-p_0,\;P(Y)-p_0,\;p_1-P(Y)\}\le PNS\le\min\{p_1,\;1-p_0,\;P(T{=}1,Y{=}1)+P(T{=}0,Y{=}0),\;p_1-p_0+P(T{=}1,Y{=}0)+P(T{=}0,Y{=}1)\}.$$
Observational data can only **tighten** the bounds.

### 2.4 Monotonicity

If treatment never harms anyone ($Y(1)\ge Y(0)$ for all units; no "harmed" type), then $PNS=p_1-p_0$, the risk difference — a fact about the joint distribution of $(Y(0),Y(1))$, not something the generic Tian–Pearl bounds discover on their own. §2.3's bounds are computed *without* assuming monotonicity, so even when the data-generating process happens to have no harmed units, the bounds computed from $p_1$, $p_0$, and the observed joint distribution alone do not automatically collapse (§3's code shows exactly this). To get the point value $PNS=p_1-p_0$, you must additionally **assume** monotonicity going in — and that assumption is itself untestable from data alone, since it is a claim about the joint distribution of $(Y(0),Y(1))$, which (Module 01's fundamental problem) is never jointly observed. §3 also shows what goes wrong if you assume monotonicity when it's actually false: the claimed value $p_1-p_0$ no longer matches the true PNS.

### 2.5 Takeaway

Interventional quantities need only graph assumptions; counterfactual quantities generally need **functional-form** or **monotonicity** assumptions. Otherwise the best one can do is report bounds.

## 3. Use It: Code

`code/pns_bounds_demo.py` simulates units with known response types and confounded treatment, computes $p_1$, $p_0$ and the observational quantities from data, applies the bounds, and checks that the true PNS lies inside them.

## Exercises

1. Set the "harmed" fraction to 0. Verify PNS equals the risk difference.
2. Make treatment unconfounded. Do the observational bounds tighten or not?
3. Why do the bounds always include the ATE-implied lower bound $p_1-p_0$?

## Key Terms

| Term | What it actually means |
|---|---|
| Response type | A unit's pair of potential outcomes $(Y(0),Y(1))$ |
| PNS | Probability that treatment is both necessary and sufficient for the outcome |
| Monotonicity | Treatment never causes harm; no unit has $Y(1)<Y(0)$ |
