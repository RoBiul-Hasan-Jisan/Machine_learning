# 03 · Optimization

Plain (mini-batch) gradient descent has real, well-documented weaknesses: it treats every parameter identically, it's slow to navigate ravines in the loss surface, and it can't tell the difference between a parameter that needs a big update and one that needs a small, careful one. Every optimizer in this lesson exists to fix a **specific, identifiable weakness** of the one before it. Understanding *why* each was introduced matters more than memorizing the update formula.

## What you'll learn, and why each exists
- **Momentum** — plain SGD oscillates in narrow ravines; momentum smooths that out
- **Exponential Moving Average (EMA)** — the statistical tool that momentum (and everything after it) is built on
- **AdaGrad** — different parameters need different learning rates; AdaGrad adapts per-parameter
- **RMSProp** — AdaGrad's learning rate shrinks to zero too aggressively; RMSProp fixes that
- **Adam** — combines Momentum's "direction memory" with RMSProp's "adaptive step size" — the default optimizer for most deep learning today
- **AdamW** — Adam's weight decay was subtly broken; AdamW fixes it
- **Learning-rate schedulers** — even the best optimizer still benefits from a well-chosen `η(t)` schedule on top

---

## The core problem with plain Gradient Descent

`W := W - η · ∂L/∂W` treats the gradient at every step as the *only* signal, and applies the *same* learning rate to every parameter. Two problems fall out of this:

1. **Ravines:** if the loss surface is much steeper in one direction than another (a common shape near a minimum), plain GD oscillates back and forth across the steep direction while crawling slowly along the shallow direction.
2. **One-size-fits-all `η`:** some parameters (e.g. those tied to rare features) might need large updates when they do get signal; others need small, careful updates. A single global `η` can't serve both well.

Every method below chips away at one of these problems.

## 1. Momentum — *fixes: oscillation in ravines*

**Why introduced:** plain SGD's updates only look at the *current* gradient, so in a ravine it zig-zags. Momentum adds "memory" of past gradients, so updates that consistently point the same way accumulate speed, while oscillating updates (which point in *different* directions each step) cancel out.

```
v := β·v + (1-β)·∂L/∂W      (velocity — an exponential moving average of past gradients)
W := W - η·v
```

`β` (typically 0.9) controls how much past gradients matter. Physically: think of a ball rolling downhill, building momentum in the direction it's consistently pushed, and less affected by small bumps.

## 2. Exponential Moving Average (EMA) — *the building block underneath everything below*

**Why it matters:** Momentum's `v` update above *is* an EMA. An EMA is simply a running average that weights recent values more heavily than old ones:

```
EMA_t = β·EMA_{t-1} + (1-β)·x_t
```

This single idea — a cheap, memory-efficient "smoothed running estimate" of a noisy quantity — is the mathematical backbone of Momentum (smoothing gradients), RMSProp/Adam (smoothing squared gradients), and even techniques like EMA of model weights for more stable final models. Once you understand EMA, every optimizer below is just "EMA applied to a different quantity."

## 3. AdaGrad — *fixes: one-size-fits-all learning rate*

**Why introduced:** parameters that receive large, frequent gradients should take smaller careful steps; parameters that rarely get gradient signal (e.g. rare vocabulary embeddings) should take larger steps when they do. AdaGrad accumulates the *sum of squared gradients* per parameter and divides the learning rate by it:

```
G := G + (∂L/∂W)²                      (running total, per parameter)
W := W - η/√(G+ε) · ∂L/∂W
```

Effect: frequently/steeply-updated parameters get their effective learning rate shrunk; rarely-updated ones keep a relatively larger effective learning rate. Good for sparse data (e.g. NLP with rare words).

**The problem AdaGrad introduced:** `G` only ever grows (it's a running *sum*, never decayed), so the effective learning rate `η/√G` monotonically shrinks toward zero — eventually training grinds to a halt, even if the model hasn't converged yet.

## 4. RMSProp — *fixes: AdaGrad's learning rate dying to zero*

**Why introduced:** replace AdaGrad's ever-growing sum with an **EMA** of squared gradients instead — old gradients "decay away" instead of accumulating forever, so the effective learning rate doesn't monotonically vanish.

```
S := β·S + (1-β)·(∂L/∂W)²              (EMA of squared gradients, not a running sum)
W := W - η/√(S+ε) · ∂L/∂W
```

This was a direct, deliberate fix: swap AdaGrad's "sum" for an EMA (the same idea from step 2), so recent gradient magnitudes matter more than ancient ones.

## 5. Adam (Adaptive Moment Estimation) — *combines Momentum + RMSProp*

**Why introduced:** Momentum gives good *direction* (smoothed gradient); RMSProp gives good *step size* (adaptive, per-parameter). Adam tracks an EMA of both the gradient itself (1st moment, like Momentum) and the squared gradient (2nd moment, like RMSProp), then combines them, with a bias-correction term because both EMAs start at zero and are biased toward zero early in training:

```
m := β₁·m + (1-β₁)·g                    (1st moment — direction, like Momentum)
v := β₂·v + (1-β₂)·g²                    (2nd moment — scale, like RMSProp)
m̂ := m / (1-β₁ᵗ)                          (bias correction)
v̂ := v / (1-β₂ᵗ)
W := W - η · m̂ / (√v̂ + ε)
```

Defaults `β₁=0.9, β₂=0.999` work well across a huge range of problems, which is a big part of why Adam became the default choice for most deep learning: it needs comparatively little tuning.

## 6. AdamW — *fixes: Adam's weight decay was subtly broken*

**Why introduced:** the common way to add L2 regularization to SGD is to add `λW` directly to the gradient before the update. In plain SGD this is mathematically equivalent to true "weight decay" (shrinking weights toward zero each step). But in Adam, adding `λW` to the gradient means it also gets divided by `√v̂` — so weight decay ends up scaled by the *adaptive* term, coupling two things that should be independent, and this measurably hurts generalization.

**AdamW's fix:** decouple weight decay from the gradient-based update entirely — apply it as a separate, direct shrinkage step:

```
W := W - η·m̂/(√v̂+ε) - η·λ·W         (weight decay applied directly, NOT through the adaptive m̂/v̂ term)
```

This single change (Loshchilov & Hutter, 2017) measurably improved generalization over Adam in many settings, and AdamW has since become the more common default in modern architectures (especially Transformers).

## 7. Learning-Rate Schedulers (on top of any optimizer)

Even Adam/AdamW benefit from scheduling `η` over time (see lesson 02's schedules — step decay, cosine annealing, warmup). Adaptive optimizers reduce, but don't eliminate, the need for a good learning-rate schedule; warmup in particular is now standard practice for training large models, since Adam's early bias-corrected estimates (`m̂`, `v̂`) can be unreliable in the first few steps.

## Quick summary: what problem did each one solve?

| Optimizer | Fixes | How |
|---|---|---|
| Momentum | Oscillation in ravines | EMA of gradients (direction memory) |
| RMSProp | AdaGrad's LR decaying to 0 | EMA (not sum) of squared gradients |
| Adam | Needing both direction memory AND adaptive step size | Combines Momentum's `m` + RMSProp's `v` |
| AdamW | Adam's weight decay being coupled to adaptive scaling | Decouples weight decay into a separate direct step |

## Run the code
[`03-optimization.ipynb`] — implements Momentum, RMSProp, Adam, and AdamW from scratch in NumPy on a small non-convex toy loss surface (so you can *see* the ravine problem and how each optimizer handles it), then reproduces the same comparison using PyTorch's `torch.optim` implementations.

