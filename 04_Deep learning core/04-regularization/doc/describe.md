# 04 · Regularization 

This is a companion to `04-regularization.md`. It fills in the mathematical
"why" behind each technique: where the penalty terms come from, what
optimization problem they're really solving, and how they connect to each
other. Read the base lesson first — this assumes you know *what* each method
does mechanically.

---

## 0. The bias–variance lens

Every regularizer is answering the same underlying question: **how do I
trade bias for variance?**

For a model $\hat f$ trained on a random training set, the expected test
error at a point $x$ decomposes as:

$$
E[(y - f̂(x))²] = Bias(f̂(x))² + Var(f̂(x)) + σ²_noise
$$



- **Bias** — error from the model being too *simple* to capture the true
  pattern (underfitting).
- **Variance** — error from the model being too *sensitive* to the
  particular training set it happened to see (overfitting). A
  high-variance model would give a very different fit if you resampled the
  training data.
- **Irreducible noise** — $\sigma^2$, the noise floor in $y$ itself.

An unregularized, high-capacity network (many parameters, trained to
convergence) tends to sit in the low-bias/high-variance regime — it fits
training data almost perfectly, but that fit is unstable across resamples.
**Every regularization technique in this lesson is a variance-reduction
tool.** It intentionally injects a small amount of bias (worse training fit)
to buy a larger reduction in variance (better, more stable test fit). This
is *the* single unifying idea — keep it in mind as we go through the math.

---

## 1. L1 / L2 as MAP estimation (the Bayesian view)

Ordinary training minimizes the negative log-likelihood of the data given
weights $W$:

$$
Ŵ_MLE = argmin_W  -log P(D | W)
$$

This is maximum likelihood estimation (MLE), and it has no mechanism to
prefer "simple" weights — it will happily drive weights to whatever values
fit the training set best, however large or specific.

**Bayesian regularization** puts a *prior* $P(W)$ on the weights and
maximizes the posterior instead:

$$
\hat{W}_{MAP}
= \arg\max_W P(W \mid D)
= \arg\max_W P(D \mid W)P(W)
$$

$$
= \arg\min_W \left[-\log P(D \mid W) - \log P(W)\right]
$$

The `-log P(W)` term is exactly your regularization penalty. Two natural
priors give you the two penalties from the base lesson:

**Gaussian prior → L2.** If each weight is assumed i.i.d. $w_i \sim
\mathcal{N}(0, \tau^2)$, then:

$$
-\log P(W)
= -\log \prod_i
\left[
\frac{1}{\sqrt{2\pi\tau^2}}
\exp\left(-\frac{w_i^2}{2\tau^2}\right)
\right]
$$

$$
= \frac{1}{2\tau^2}\sum_i w_i^2 + \text{const}
$$

That's a constant times $\Sigma w_i^2$ — precisely the L2 penalty, with
$\lambda = 1/2\tau^2$. A **small prior variance $\tau^2$** (strong belief
that weights are near zero) corresponds to a **large $\lambda$** (strong
regularization). This is why L2 is often called "weight decay under a
Gaussian prior."

**Laplace prior → L1.** If instead $w_i \sim \text{Laplace}(0, b)$, with
density $\frac{1}{2b}\exp(-|w_i|/b)$:

$$
-log P(W) = (1/b)·Σᵢ |wᵢ|  +  const
$$

— exactly the L1 penalty, $\lambda = 1/b$. The Laplace distribution has a
sharp peak (non-differentiable cusp) at zero and heavier tails than a
Gaussian. That cusp is *why* L1 produces exact zeros: the prior is telling
the optimizer "I believe most weights are exactly zero, with a few large
exceptions," which is a fundamentally different structural assumption than
the Gaussian's "all weights are small-ish."

This reframes the whole lesson: **regularization = encoding a prior belief
about what a "reasonable" weight looks like, then finding the MAP estimate
instead of the MLE.**

---

## 2. Penalty form vs. constraint form (Lagrangian duality)

The penalized objective:

$$
minimize_W   L(W) + λ·R(W)
$$

is the *Lagrangian relaxation* of a constrained problem:

$$
minimize_W   L(W)
$$

$$
subject to   R(W) ≤ t
$$


for some budget $t$ that's a decreasing function of $\lambda$ (Lagrange
multiplier duality — for convex $L$ and $R$, every $\lambda \geq 0$
corresponds to some $t \geq 0$ and the two problems have the same
solution). This is worth sitting with because it makes the geometry of
L1-vs-L2 sparsity concrete:

- **L2 constraint** $\Sigma w_i^2 \leq t$ describes a **ball** (smooth,
  round) in weight space.
- **L1 constraint** $\Sigma |w_i| \leq t$ describes a **cross-polytope**
  (a diamond in 2D, an octahedron in 3D) — a shape with sharp *corners*
  sitting exactly on the coordinate axes.

The unconstrained loss $L(W)$ has elliptical contours in weight space, and
the constrained solution is the point where the smallest such contour
first touches the constraint region. Because the L1 polytope's corners lie
*on the axes* (i.e., at points where some $w_i = 0$), contours are
disproportionately likely to first touch at a corner — producing an exact
zero. The L2 ball has no preferred contact point along the axes, so the
touching point is generically some smooth combination of all coordinates,
none exactly zero. This is the geometric version of the gradient argument
in the base lesson (constant vs. shrinking gradient near zero) — same
conclusion, different lens.

---

## 3. Weight decay: SGD vs. Adam, worked out

Base lesson claims L2-regularization and weight-decay are equivalent for
SGD but *not* for Adam. Here's the derivation.

**Plain SGD.** L2-regularized loss is $L_{tot} = L + \frac{\lambda}{2}\|W\|^2$.
Its gradient adds $\lambda W$ to the raw gradient $g = \nabla L$:

$$
W ← W - η(g + λW) = W - ηg - ηλW
$$

That's exactly "take a normal SGD step, then shrink $W$ by factor
$(1-\eta\lambda)$" — i.e. decoupled weight decay. **They're identical for
SGD.**

**Adam.** Adam doesn't apply the raw gradient directly — it rescales each
coordinate by a running estimate of its second moment:

$$
m ← β₁m + (1-β₁)g
$$

$$
v ← β₂v + (1-β₂)g²
$$

$$
W ← W - η · m/(√v + ε)
$$


If you fold the L2 penalty into $g$ (i.e. $g \to g + \lambda W$) *before*
this, the $\lambda W$ term gets divided by $\sqrt{v}+\varepsilon$ along with
everything else. Since $v$ (and hence the effective step size) differs
**per parameter** and changes over training, the amount of actual shrinkage
applied to each weight becomes coupled to that weight's gradient history —
weights with small historical gradients get *disproportionately large*
decay relative to their gradient signal, and vice versa. This is an
accidental, uncontrolled side-effect, not the intended "shrink every
weight by a fixed proportion" behavior.

**AdamW's fix** is to apply the decay *outside* Adam's normalization
entirely:

$$
W ← W - η·m/(√v + ε) - η·λ·W        (decay term added after, unscaled by v)
$$

This restores the clean "shrink every weight by the same proportion each
step" semantics, decoupled from the adaptive learning rate — hence
*decoupled* weight decay.

---

## 4. Dropout as implicit ensembling — the math

Dropout's "why it works" story is usually stated informally ("prevents
co-adaptation"). Here's the more precise version.

For a layer with $n$ units, applying dropout with keep-probability $1-p$
per forward pass samples one of $2^n$ possible "thinned" sub-networks
(each unit present or absent). Training with dropout over many steps
approximates training an **exponential ensemble** of these $2^n$
sub-networks, with extensive weight sharing between them (since they all
draw from the same underlying weight matrix).

At inference, we don't actually average $2^n$ networks' outputs (that's
intractable). Instead we use the full network with **no** dropout, but
with each unit's *outgoing* weights scaled to match its expected value
during training. If a unit was kept with probability $1-p$ during
training, its expected contribution to the next layer was $(1-p)\cdot a$
where $a$ is its full activation. Two equivalent ways to match that at
inference:

- **Scale at test time:** multiply activations by $(1-p)$ at inference
  (the classical formulation), or
- **Inverted dropout** (what the base lesson's pseudocode uses): scale by
  $1/(1-p)$ *during training* instead, so inference needs no
  modification at all — this is what virtually all modern frameworks
  implement, because it keeps the inference graph identical to a
  dropout-free network.

This inference-time rescaled full network is a first-order (single
forward pass) approximation to the true ensemble average — known as the
**weight-scaling rule**, and it's provably exact for a single linear
layer, and empirically a very good approximation for the deep nonlinear
case.

**A second, complementary view:** dropout is equivalent (in expectation,
to first order) to an L2-type penalty that scales with each unit's
squared activation and the layer's dropout rate — i.e., it's not *only*
an ensembling trick, it also has an adaptive, data-dependent regularizing
effect similar in spirit to L2, but applied to activations rather than
weights directly.

---

## 5. Early stopping ≈ L2 regularization (for quadratic loss)

This is a classical (Bishop-style) result that ties early stopping back
to L2, closing the loop between two seemingly unrelated techniques.

Consider a quadratic loss (a 2nd-order Taylor expansion of $L(W)$ around
the unconstrained minimizer $W^*$), and diagonalize the Hessian $H$ via
its eigendecomposition. Gradient descent with learning rate $\eta$, after
$\tau$ steps, shrinks each eigen-direction $i$ toward $W_i^*$ by a factor
that depends on that direction's eigenvalue $h_i$ and on $\tau$:

$$
shrinkage factor ≈ 1 - (1 - ηh_i)^τ
$$

Compare this to the closed-form solution of **L2-regularized** least
squares, whose shrinkage factor in the same eigenbasis is:

$$
shrinkage factor = h_i / (h_i + λ)
$$

Both expressions do the same qualitative thing: **shrink low-curvature
("flat") directions more aggressively than high-curvature ("steep")
directions**, relative to the unconstrained optimum. In the early-stopping
formula, fewer steps $\tau$ (or smaller $\eta$) behaves like *larger*
$\lambda$; more steps behaves like *smaller* $\lambda$ (less
regularization, approaching the unregularized MLE as $\tau\to\infty$).
This is why early stopping and L2 are often described as approximately
interchangeable for quadratic/near-quadratic loss surfaces — **the number
of training steps itself acts as an inverse regularization strength.**

---

## 6. Batch normalization — the actual gradient, and why it complicates the "regularizer" story

For completeness, the backward pass through a batchnorm layer (batch size
$m$, per-feature) requires differentiating through the batch statistics
themselves, since $\mu_B$ and $\sigma_B^2$ are *functions of every example
in the batch*:

$$
\frac{\partial L}{\partial \hat{x}_i}= \frac{\partial L}{\partial y_i}\gamma
$$

$$
\frac{\partial L}{\partial \sigma_B^2}=
\sum_i
\frac{\partial L}{\partial \hat{x}_i}
(x_i-\mu_B)
\left(-\frac{1}{2}\right)
(\sigma_B^2+\epsilon)^{-3/2}
$$

$$
\frac{\partial L}{\partial \mu_B}=
\sum_i
\frac{\partial L}{\partial \hat{x}_i}
\left(-\frac{1}{\sqrt{\sigma_B^2+\epsilon}}\right)
+
\frac{\partial L}{\partial \sigma_B^2}
\sum_i
\frac{-2(x_i-\mu_B)}{m}
$$

$$
\frac{\partial L}{\partial x_i}=
\frac{\partial L}{\partial \hat{x}_i}
\frac{1}{\sqrt{\sigma_B^2+\epsilon}}
+
\frac{\partial L}{\partial \sigma_B^2}
\frac{2(x_i-\mu_B)}{m}
+
\frac{\partial L}{\partial \mu_B}
\frac{1}{m}
$$


The key structural fact buried in this: **the gradient with respect to
any single example $x_i$ depends on every other example in the batch**
(through $\mu_B$ and $\sigma_B^2$). This is *exactly* the mechanism behind
its regularizing side-effect — each example's effective loss landscape is
being perturbed by whichever other examples happened to land in its
minibatch, which changes randomly every epoch. It's also why batch norm's
behavior is **batch-size dependent** (small batches → noisier $\mu_B,
\sigma_B^2$ → more regularization but less stable training statistics),
and why it needs a different formulation at inference (running averages
of $\mu_B, \sigma_B^2$ accumulated over training, since there's no "batch"
at inference time for a single query).

---

## 7. Why these all "work": a capacity-control unification

Formal learning theory (VC dimension, Rademacher complexity) bounds the
gap between training error and test error as an increasing function of a
model's **effective capacity** — roughly, how many different labelings
of the training data the model *could* fit if it needed to. Every
technique above reduces effective capacity without reducing the *raw*
parameter count:

| Technique | How it reduces effective capacity |
|---|---|
| L1 / L2 | Shrinks the *usable range* of each weight (small-norm hypothesis space) |
| Dropout | Forces robustness to any single unit's removal → shared-representation capacity, not $2^n$ independent sub-networks |
| Early stopping | Restricts optimization to weights reachable in $\tau$ steps from initialization — a strictly smaller reachable set than the full unconstrained optimum |
| Data augmentation | Doesn't shrink the hypothesis space directly — instead enlarges the *effective* dataset, which shrinks the generalization gap for a fixed hypothesis space |
| Batch norm | Adds training-time stochastic noise, functioning similarly to a soft, batch-size-dependent penalty |

This is the thread connecting the "what" (base lesson) to the "why"
(this doc): raw parameter count is a poor proxy for a model's true
capacity to overfit, and every regularizer is a different lever for
controlling *effective* capacity instead.

## Suggested follow-up reading
- Bishop, *Pattern Recognition and Machine Learning*, Ch. 3.3 (Bayesian
  linear regression) and Ch. 5.5.2 (early stopping as regularization)
- Srivastava et al. 2014, "Dropout: A Simple Way to Prevent Neural
  Networks from Overfitting" (the weight-scaling approximation is proven
  here for the linear case)
- Loshchilov & Hutter 2017, "Decoupled Weight Decay Regularization"
  (AdamW) — the SGD/Adam mismatch derivation in full
- Ioffe & Szegedy 2015, "Batch Normalization" — original internal
  covariate shift argument, plus later critiques (Santurkar et al. 2018)
  arguing the loss-landscape-smoothing effect matters more than
  covariate shift
