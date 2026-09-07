# 02 · Training Neural Networks

This is a deeper, more mathematical companion to the original lesson. It keeps the same six sections but adds the formal reasoning behind each practical rule: why mini-batches work, why learning rate has the effect it does, and how convergence guarantees are actually derived.

---

## 0. Setup and Notation

We're minimizing an empirical loss over a dataset of $N$ examples:

$$
L(W) = \frac{1}{N}\sum_{i=1}^{N} \ell(f(x_i; W), y_i)
$$

where $f(x_i; W)$ is the network's prediction, $\ell$ is a per-example loss (e.g. cross-entropy), and $W$ is the full parameter vector.

Gradient descent updates parameters using the gradient of $L$:

$$
W_{t+1} = W_t - \eta \nabla L(W_t)
$$

Everything in this lesson is really about **how we approximate $\nabla L(W_t)$** and **how big $\eta$ should be**.

---

## 1. Batch Gradient Descent — Formal View

Batch GD computes the *exact* gradient:

$$
\nabla L(W) = \frac{1}{N}\sum_{i=1}^N \nabla \ell(f(x_i; W), y_i)
$$

**Why it's stable:** this is an unbiased, zero-variance estimate of the true gradient (it *is* the true gradient). If $L$ is convex and $\eta$ is small enough, batch GD is guaranteed to decrease $L$ monotonically every step.

### Convergence rate (convex, $\eta_t$-Lipschitz-smooth $L$):

$$
\sum_{t=1}^{\infty} \eta_t = \infty,
\qquad
\sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

(e.g. $\eta_t = c/t$ satisfies both). The first condition ensures you can travel
arbitrarily far if needed; the second ensures the accumulated noise variance
stays finite so you actually settle down.

This is an $O(1/t)$ rate — to halve your error you need roughly twice as many iterations. For strongly convex $L$, this improves to a linear rate $O(\rho^t)$ for some $\rho < 1$. Neural network losses are non-convex, so these bounds don't strictly apply, but they explain the *qualitative* behavior: batch GD's steps are well-behaved because the gradient direction is trustworthy.

**Cost per step:** $O(N)$ forward/backward passes. This is the real reason it doesn't scale — not memory alone, but that a single parameter update costs an entire epoch's worth of compute.

---

## 2. Stochastic Gradient Descent — Bias, Variance, and Noise

SGD replaces the true gradient with a single-example estimate:

$$
g_t = \nabla \ell(f(x_{i_t}; W_t), y_{i_t}), \quad i_t \sim \text{Uniform}(1, \dots, N)
$$

**Key fact — SGD is unbiased:**

$$
\mathbb{E}_{i_t}[g_t] = \frac{1}{N}\sum_{i=1}^N \nabla \ell(f(x_i; W_t), y_i) = \nabla L(W_t)
$$

So *on average* SGD moves in the right direction — but any single step can be wildly wrong. Formally, we can decompose the mean squared error of the estimate:

$$
\mathbb{E}\left[\|g_t - \nabla L(W_t)\|^2\right] = \text{Var}(g_t)
$$

Since $g_t$ is unbiased, all of its error is variance, not bias. This variance is exactly the "noise" you see in a jagged SGD loss curve.

**Why the noise can help:** near a saddle point or shallow local minimum, the true gradient $\nabla L(W_t)$ is near zero, so batch GD stalls. SGD's per-step gradient is *not* near zero (it's a noisy single-example estimate), so it can kick the parameters out of flat regions. This is sometimes modeled as SGD performing gradient descent on $L$ plus an implicit random-walk term — related to viewing SGD as discretized Langevin dynamics.

**Convergence rate:** because of the variance, SGD with a *fixed* learning rate does not converge to $W^*$ exactly — it converges to a noise ball around it, of size roughly proportional to $\eta \cdot \text{Var}(g_t)$. To actually converge, $\eta$ must be *decayed* over time (this is the classical Robbins–Monro condition):



---

## 3. Mini-Batch SGD — Variance Reduction by Averaging

A mini-batch of size $B$ averages $B$ independent per-example gradient estimates:

$$
g_t^{(B)} = \frac{1}{B}\sum_{j=1}^{B} \nabla \ell(f(x_{i_j}; W_t), y_{i_j})
$$

Since each term has the same variance $\sigma^2 = \text{Var}(g_t)$ (single-example gradient variance) and the samples are drawn independently:

$$
\text{Var}(g_t^{(B)}) = \frac{\sigma^2}{B}
$$

**This is the central mathematical fact behind mini-batching:** variance shrinks proportional to $1/B$, so the standard deviation (the "noise" you actually see in the loss curve) shrinks proportional to $1/\sqrt{B}$. Doubling the batch size only reduces gradient noise by a factor of $\sqrt{2}$, not 2 — which is why batch sizes give diminishing returns and why very large batches aren't simply "better."

**The compute/noise tradeoff:** a batch of size $B$ costs $B\times$ the compute of one example but only reduces noise by $\sqrt{B}\times$. This is the formal reason there's a sweet spot: too small and updates are noisy and hardware-inefficient; too large and you're paying $B$ compute for diminishing noise reduction, while also taking fewer total steps per epoch (fewer opportunities to update).

**Linear scaling rule (empirical heuristic derived from this theory):** if you increase batch size by a factor $k$, you can roughly increase $\eta$ by the same factor $k$ to keep the *effective* step noise and per-epoch progress comparable — because the reduced gradient variance can tolerate a proportionally larger step.

---

## 4. Learning Rate — Why "Too High" Diverges and "Too Low" Crawls

Assume $L$ is $\beta$-smooth, meaning its gradient doesn't change too fast:

$$
\|\nabla L(W_1) - \nabla L(W_2)\| \le \beta \|W_1 - W_2\| \quad \text{for all } W_1, W_2
$$

This is equivalent to saying the loss surface's curvature is bounded above by $\beta$ (think of $\beta$ as an upper bound on the largest eigenvalue of the Hessian). A standard descent lemma then gives:

$$
L(W_{t+1}) \le L(W_t) - \eta\left(1 - \frac{\beta \eta}{2}\right)\|\nabla L(W_t)\|^2
$$

Look at the coefficient $\left(1 - \frac{\beta\eta}{2}\right)$:

- If $\eta < \frac{1}{\beta}$, the coefficient is positive — every step is *guaranteed* to decrease the loss (for convex/smooth $L$). This is where "just right" comes from mathematically.
- If $\eta = \frac{2}{\beta}$, the coefficient hits zero — steps stop making guaranteed progress.
- If $\eta > \frac{2}{\beta}$, the coefficient goes negative — the bound *guarantees the loss can increase*, which is the divergence/oscillation you see when $\eta$ is too high.

**Too low, formally:** the guaranteed per-step decrease is $\eta(1-\beta\eta/2)\|\nabla L(W_t)\|^2$ — this is linear in $\eta$, so halving $\eta$ roughly halves your guaranteed progress per step, meaning you need roughly twice as many steps for the same total progress. This is the "crawls" behavior.

**Why $1/\beta$ shows up everywhere:** $\beta$ is essentially the curvature of the loss surface. A learning rate near $1/\beta$ is the largest step you can take before you start "overshooting" the local quadratic bowl the gradient is describing — which is precisely the intuition of overshooting the minimum.

---

## 5. Epoch, Batch, Iteration — as a Sampling Process

Formally, mini-batch SGD implements sampling **without replacement within an epoch**: the dataset is shuffled and partitioned into $N/B$ disjoint batches, so over one epoch every example is used exactly once. This is slightly different from the "true" i.i.d. sampling assumed in the SGD variance analysis above, but is close enough in practice, and it guarantees an even coverage of the dataset per epoch.

$$
\text{iterations per epoch} = \left\lceil \frac{N}{B} \right\rceil, \qquad \text{total updates} = \text{epochs} \times \left\lceil \frac{N}{B} \right\rceil
$$

This matters because learning-rate schedules (below) are usually defined as a function of the **iteration index** $t$, not the epoch — so the same schedule formula produces a different actual trajectory depending on batch size, since batch size determines how many $t$'s occur per epoch.

---

## 6. Learning-Rate Scheduling — the Math Behind Each Rule

Recall the descent lemma from Section 4: the "safe" learning rate depends on $\beta$, the local curvature of the loss surface. The core reason schedules help is that **$\beta$ effectively changes over the course of training** — the loss surface is not one fixed quadratic bowl. Early in training (far from any minimum, near random initialization) the useful step size for fast progress can be relatively large; near a minimum the local curvature is often sharper along some directions, so a large $\eta$ risks the divergence condition from Section 4.

- **Step decay:** $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/N_{\text{step}} \rfloor}$ for decay factor $\gamma$ (e.g. 0.1) every $N_{\text{step}}$ epochs. Piecewise-constant approximation to "shrink $\eta$ as you approach the optimum."

- **Exponential decay:** $\eta(t) = \eta_0 e^{-kt}$. Smooth, continuous version of the same idea; $k$ controls how fast $\eta$ shrinks.

- **Cosine annealing:** $\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$ over a total of $T$ steps. Spends more time at both the high end and near-zero end (cosine's slope is flattest at the endpoints) and moves through the middle range faster than linear decay — empirically this "long plateau near zero at the end" behavior helps fine-tune convergence.

- **Warmup:** ramps $\eta$ linearly from ~0 up to $\eta_0$ over the first $T_{\text{warmup}}$ steps: $\eta_t = \eta_0 \cdot t / T_{\text{warmup}}$. Justification ties back to Section 4: at random initialization, gradients (and therefore the effective local $\beta$) can be very large and poorly estimated, especially with adaptive optimizers whose second-moment estimates haven't stabilized yet — so starting small avoids taking a step past the "safe" $1/\beta$ threshold before that threshold is even well estimated.

- **ReduceLROnPlateau:** rather than following a fixed function of $t$, this shrinks $\eta$ (by factor $\gamma$) whenever a monitored quantity (e.g. validation loss) fails to improve by more than a threshold for a "patience" window of $P$ epochs. This is a *feedback-controlled* schedule — it responds to evidence that the optimizer has entered the regime described at the end of Section 4 (guaranteed-progress term $\to 0$) rather than assuming a fixed timetable for it.

---

## Summary Table

| Concept | Formal quantity | Practical consequence |
|---|---|---|
| Batch GD | Zero-variance, exact gradient | Slow per step, monotonic decrease |
| SGD | Unbiased, high-variance gradient | Noisy but cheap; noise helps escape saddle points |
| Mini-batch | Variance $\propto 1/B$ | Noise shrinks as $\sqrt{B}$; diminishing returns |
| Learning rate | Governed by smoothness constant $\beta$ | Safe range roughly $\eta < 1/\beta$ |
| Schedules | Time-varying $\eta_t$ | Compensate for changing effective curvature over training |

## Suggested next step
Pair this with the notebook's loss-curve comparisons: try plotting the *variance* of the gradient estimate itself (not just the loss) across Batch GD / SGD / Mini-Batch at a few batch sizes, and check empirically whether it scales like $1/B$ as predicted in Section 3.
