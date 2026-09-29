# 39. Causal Deep Learning

## Learning Objectives

- Explain why standard deep nets trained by empirical risk minimization (ERM) exploit spurious correlations and fail under distribution shift
- Describe invariant risk minimization (IRM) as a search for a representation whose optimal predictor is the same across environments
- Implement a TARNet-style neural network for CATE estimation, connecting it back to the T-learner (Module 25)
- Recognize the trade-offs: IRM's difficulty in practice, and neural CATE's need for the same identification assumptions as every earlier method

---

## 1. The Problem

Deep networks are powerful function approximators, but ERM on i.i.d. training data has no reason to prefer a network that uses a variable's *causal* relationship to the label over one that merely uses a correlated, spurious shortcut (e.g., background pixels correlated with an animal class in the training set). This module looks at two ways causal thinking has been brought into deep learning: (i) learning representations that are robust to spurious shortcuts, and (ii) building networks whose *architecture* encodes a causal estimand.

## 2. The Concept

### 2.1 Spurious correlations and environments

Suppose training data comes from environments $e\in\mathcal E$ (hospitals, camera setups, time periods) where a spurious feature $X_{spu}$ correlates with $Y$ differently in each environment, while a causal feature $X_{cau}$ predicts $Y$ the *same* way everywhere. ERM pools all environments and may lean on $X_{spu}$ if it is a stronger predictor pooled across the (limited) training environments — and then fail on a new environment where that correlation reverses.

### 2.2 Invariant Risk Minimization (Arjovsky et al., 2019)

IRM looks for a representation $\Phi(X)$ such that a *single* classifier $w$ on top of $\Phi$ is simultaneously optimal in **every** training environment:
$$\min_{\Phi,w}\ \sum_{e\in\mathcal E}R^e(w\circ\Phi)\quad\text{s.t. } w\in\arg\min_{\bar w}R^e(\bar w\circ\Phi)\ \ \forall e.$$
The intuition, connecting back to Module 38: if $\Phi$ only keeps the causal feature (whose relationship to $Y$ is invariant across environments, by the definition of "causal" used here), a single $w$ can be optimal everywhere; a $\Phi$ that keeps the spurious feature cannot, because the optimal way to use it changes across environments. The invariance constraint is a proxy for "$\Phi$ captures only causal parents of $Y$", under specific structural assumptions that do not always hold in practice, and the practical (IRMv1) relaxation of the constraint is known to be hard to optimize reliably.

### 2.3 TARNet: a neural T-learner

TARNet (Shalit, Johansson, Sontag, 2017) shares a representation network $\Phi(X)$ across both arms, then splits into two head networks $h_1,h_0$ predicting $Y$ under each treatment:
$$\hat Y(t)=h_t(\Phi(X)).$$
This is architecturally a T-learner (Module 25) with a shared body — sharing $\Phi$ lets both heads borrow statistical strength from the same learned features, while still letting each head fit an arm-specific response surface. Extensions add a representation-balancing penalty (an integral probability metric between treated and control representations) motivated by generalization bounds, but the core identification assumptions (unconfoundedness, positivity, Module 06) are unchanged from every non-neural method in this course — a bigger network buys flexibility, not new identifying power.

### 2.4 What deep learning changes, and what it doesn't

Deep architectures can represent complex $\mu_t(x)$ or $\Phi$ functions and scale to unstructured data (images, text) other estimators cannot use directly. They do **not** relax the need for unconfoundedness, positivity, or a defensible causal graph — Module 06's assumptions, and Module 17's identification logic, apply just as much to a transformer as to a linear regression.

## 3. Use It: Code

`code/tarnet_demo.py` implements a small TARNet-style two-head MLP (NumPy, manual forward/backward for a single hidden layer, to stay dependency-light) on simulated heterogeneous-effect data, compares its ATE and per-unit effect estimates to a plain T-learner with separate networks (no shared body), and to the truth.

## Exercises

1. Increase the hidden-layer width. Does the shared-body TARNet or the fully separate T-learner benefit more, and why might that depend on how similar the two response surfaces are?
2. Sketch (in words) a spurious-correlation dataset with two environments where ERM would fail and explain what an invariant representation would need to discard.
3. Why doesn't a more powerful neural architecture help if the true confounder is never measured?

## Key Terms

| Term | What it actually means |
|---|---|
| Invariant Risk Minimization | Learning a representation whose optimal predictor is the same across training environments, as a proxy for capturing only causal features |
| Spurious correlation | An association that holds in the training distribution but is not stable across environments/interventions |
| TARNet | A shared-representation, two-head neural network for estimating potential outcomes under each treatment |
