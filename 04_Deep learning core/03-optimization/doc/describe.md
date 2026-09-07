# 03 · Optimization 

A deeper, more mathematical companion to the optimization lesson. Same structure, same "what problem does this solve" framing — but with the math that makes each fix precise rather than just plausible.

---

## 0. Reframing the Core Problem: Curvature and Conditioning

Near a local minimum, a twice-differentiable loss can be approximated by its second-order Taylor expansion:

$$
L(W) \approx L(W^*) + \frac{1}{2}(W-W^*)^\top H (W-W^*)
$$

where $H = \nabla^2 L(W^*)$ is the Hessian. The eigenvalues of $H$, $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n > 0$, describe the curvature along each principal direction. The **condition number**

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

is the single number that predicts how badly plain GD will oscillate. Along the steepest direction ($\lambda_{\max}$), the safe step size (Lesson 02, Section 4) is roughly $1/\lambda_{\max}$; along the shallowest direction ($\lambda_{\min}$), useful progress needs a step closer to $1/\lambda_{\min}$. A single global $\eta$ has to satisfy the *stricter* (smaller) bound $\eta < 2/\lambda_{\max}$, which makes progress along the shallow direction painfully slow whenever $\kappa \gg 1$. This is the precise mathematical definition of a "ravine": high $\kappa$.

Every optimizer in this lesson is, underneath, an attempt to make the *effective* curvature the optimizer experiences closer to $\kappa \approx 1$ (well-conditioned), either by smoothing the gradient's direction (Momentum) or rescaling each coordinate individually (AdaGrad/RMSProp/Adam).

---

## 1. Momentum — Formal Analysis

The momentum update is:

$$
v_t = \beta v_{t-1} + (1-\beta) g_t, \qquad W_t = W_{t-1} - \eta v_t
$$

**Unrolling the recursion** shows $v_t$ is a weighted sum of *all* past gradients, with exponentially decaying weights:

$$
v_t = (1-\beta)\sum_{k=0}^{t} \beta^{k} g_{t-k}
$$

**Why it kills oscillation (eigenvector decomposition):** decompose the gradient into the loss surface's principal directions. Along a **high-curvature direction**, consecutive gradients alternate in sign (the classic zig-zag), so the terms in the sum above partially cancel — momentum averages them toward zero, damping the oscillation. Along a **low-curvature direction**, consecutive gradients point the *same* way, so the terms constructively add, and $v_t$ can grow much larger than any single $g_t$ — effectively **amplifying** the step size in the direction that needed it most. In the steady state (constant gradient $g$), momentum's effective step size approaches:

$$
\|v_\infty\| = \|g\| \quad \text{(same as raw gradient — momentum only reweights the } (1-\beta)\text{ normalization here as written)}
$$

but the more common way people state momentum's benefit is via its *acceleration* interpretation: for quadratic objectives, momentum (specifically Polyak's heavy-ball method) achieves a convergence rate of

$$
O\left(\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t\right)
$$

compared to plain GD's

$$
O\left(\left(\frac{\kappa-1}{\kappa+1}\right)^t\right)
$$

Since $\sqrt\kappa < \kappa$ for $\kappa>1$, momentum's rate constant is strictly better — this is the formal version of "momentum navigates ravines faster."

---

## 2. Exponential Moving Average — Bias and the Correction Term

An EMA $E_t = \beta E_{t-1} + (1-\beta)x_t$, initialized at $E_0 = 0$, unrolls to:

$$
E_t = (1-\beta)\sum_{k=0}^{t-1}\beta^k x_{t-k}
$$

Taking the expectation, assuming $x_t \approx x$ is roughly stationary:

$$
\mathbb{E}[E_t] = (1-\beta^t)\, x
$$

**This is the source of Adam's bias-correction term.** Since $\mathbb{E}[E_t] = (1-\beta^t)x \ne x$, the raw EMA systematically *underestimates* $x$ early in training (when $t$ is small and $\beta^t$ is still close to 1). Dividing by $(1-\beta^t)$ exactly cancels this factor:

$$
\hat{E}_t = \frac{E_t}{1-\beta^t} \implies \mathbb{E}[\hat{E}_t] = x
$$

— this is precisely Adam's $\hat m = m/(1-\beta_1^t)$ and $\hat v = v/(1-\beta_2^t)$. Without this correction, Adam's early steps would be artificially small (since both $m$ and $v$ start at 0 and are biased toward 0), which is exactly why the correction matters *most* in the first few dozen/hundred iterations and becomes negligible as $\beta^t \to 0$.

---

## 3. AdaGrad — Why Per-Coordinate Scaling Fixes Conditioning

AdaGrad's update, written per-coordinate $i$:

$$
G_{t,i} = \sum_{k=1}^{t} g_{k,i}^2, \qquad W_{t,i} = W_{t-1,i} - \frac{\eta}{\sqrt{G_{t,i}}+\epsilon}\, g_{t,i}
$$

**Why this attacks $\kappa$ directly:** a coordinate that consistently receives large gradients (steep direction) accumulates a large $G_i$, shrinking its effective learning rate $\eta/\sqrt{G_i}$; a coordinate with small, infrequent gradients (shallow direction) keeps $G_i$ small, keeping its effective learning rate closer to $\eta$. This is a *diagonal preconditioner* — an approximation to dividing by the Hessian's diagonal, which is exactly the classical fix for ill-conditioning (compare to Newton's method, which divides by the full Hessian). AdaGrad has strong theoretical guarantees in the **convex, especially sparse-gradient**, setting: its regret bound is

$$
\text{Regret}_T = O\left(\sqrt{T}\right) \text{ in the worst case, but as good as } O(\log T) \text{ for sparse gradients}
$$

**The formal failure mode:** $G_{t,i}$ is a *monotonically non-decreasing sum*. As $t\to\infty$, $G_{t,i}\to\infty$ for any coordinate that ever receives nonzero gradient, so $\eta/\sqrt{G_{t,i}} \to 0$. In a non-convex, non-sparse deep learning setting (where gradients keep arriving throughout training, not just early on), this drives the effective learning rate to zero well before convergence — training doesn't diverge, it just stalls.

---

## 4. RMSProp — EMA as a Fix to a Monotonicity Problem

RMSProp replaces the cumulative sum with an EMA:

$$
S_t = \beta S_{t-1} + (1-\beta) g_t^2, \qquad W_t = W_{t-1} - \frac{\eta}{\sqrt{S_t}+\epsilon} g_t
$$

**Why this specifically fixes AdaGrad:** $S_t$ is now a weighted average of only the *last* $\sim \frac{1}{1-\beta}$ squared gradients (this "effective window" falls out of the same geometric-decay argument as Section 2). If gradient magnitudes shrink later in training (as they typically do near a minimum), $S_t$ can shrink too — the effective learning rate can go back **up** if needed, something AdaGrad's monotone sum structurally cannot do. In the stationary-noise regime, $S_t$ converges toward $\mathbb{E}[g^2]$ for the current region of the loss surface, so $\eta/\sqrt{S_t}$ tracks a *local* estimate of gradient scale rather than an all-time cumulative one.

---

## 5. Adam — Combining the Two Fixes, and Why the Combination Is Non-Trivial

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad\text{(direction: same math as Momentum)}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad\text{(scale: same math as RMSProp)}
$$
$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$
$$
W_t = W_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

**Why this isn't just "Momentum + RMSProp glued together" trivially:** the two moments track different statistics of the *same* underlying noisy signal $g_t$ — $m_t$ tracks $\mathbb{E}[g_t]$ (the signal), $v_t$ tracks $\mathbb{E}[g_t^2]$ (signal² + noise variance). The ratio $\hat m_t / \sqrt{\hat v_t}$ is, loosely, an estimate of a **signal-to-noise ratio**: for a coordinate whose gradient is consistently pointing one direction (high $|m_t|$ relative to $\sqrt{v_t}$), the update stays close to full size; for a coordinate whose gradient is mostly noise (small $m_t$, large $v_t$ from oscillating sign), the ratio shrinks toward zero automatically. This is a more refined behavior than either ingredient alone: Momentum alone doesn't rescale per-coordinate; RMSProp alone doesn't smooth *direction*, only magnitude.

**Default hyperparameters, why they work broadly:** $\beta_1=0.9$ gives $m_t$ an effective averaging window of $\sim 1/(1-0.9)=10$ steps (direction memory); $\beta_2=0.999$ gives $v_t$ a much longer window of $\sim 1000$ steps (a slowly-changing estimate of gradient scale). This asymmetry — short memory for direction, long memory for scale — is deliberate: direction should adapt quickly to the current local landscape, while the *typical magnitude* of gradients is a more slowly-varying property of the current region of parameter space.

---

## 6. AdamW — Making the Failure Mode Precise

Consider ℓ2-regularized loss $\tilde L(W) = L(W) + \frac{\lambda}{2}\|W\|^2$, whose gradient is $\nabla \tilde L = g + \lambda W$.

**Plain SGD with this regularized gradient:**

$$
W_t = W_{t-1} - \eta(g_{t-1} + \lambda W_{t-1}) = (1-\eta\lambda)W_{t-1} - \eta g_{t-1}
$$

This *is* true weight decay: every step, $W$ is multiplied by $(1-\eta\lambda)$ — a clean, uniform shrinkage independent of the gradient's magnitude.

**Adam with the same regularized gradient** $g_t' = g_t + \lambda W_{t-1}$ substituted into $m_t, v_t$:

$$
v_t \approx \beta_2 v_{t-1} + (1-\beta_2)(g_t + \lambda W_{t-1})^2
$$

The $\lambda W_{t-1}$ term now contributes to $v_t$, and the final update divides by $\sqrt{\hat v_t}$ — so the *effective* decay applied to $W$ becomes $\eta\lambda / \sqrt{\hat v_t}$, which is **different for every coordinate** and changes over training, rather than being the clean, uniform $\eta\lambda$ that regularization is supposed to provide. Parameters with large $\hat v_t$ (frequently/steeply updated) get *less* decay than intended; parameters with small $\hat v_t$ get *more*. This coupling is exactly the bug.

**AdamW's fix**, written explicitly:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1}+(1-\beta_2)g_t^2 \quad\text{(no } \lambda W \text{ mixed in here)}
$$
$$
W_t = W_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda W_{t-1}\right)
$$

The weight-decay term $\lambda W_{t-1}$ is added *after* the adaptive division, so it always applies as a clean, uniform multiplicative shrinkage $(1-\eta\lambda)$, decoupled from the per-coordinate adaptive scaling — restoring the same clean decay behavior plain SGD had.

---

## 7. Schedulers on Top of Adaptive Optimizers — Why Warmup Matters More Here

At $t$ small, $v_t$ is estimated from very few samples, so $\hat v_t$ (even after bias correction) can have **high variance** as an estimate of the true second moment — a few unlucky large early gradients can produce a misleadingly large or small $\hat v_t$, making the ratio $\hat m_t/\sqrt{\hat v_t}$ unreliable in exactly the way Lesson 02's smoothness-constant argument (Section 4 there) warns against: an unreliable local curvature estimate makes *any* fixed $\eta$ potentially unsafe. Warmup — starting $\eta$ near zero and ramping up — buys time for $v_t$'s estimate to stabilize (i.e., for $t$ to be large enough that the effective averaging window of $\beta_2$ has been meaningfully filled) before allowing large steps.

---

## Summary: The Curvature-Fixing Lens

| Optimizer | Effective fix in terms of conditioning $\kappa$ |
|---|---|
| Momentum | Damps oscillation along high-curvature directions, amplifies progress along low-curvature ones — improves the *effective* convergence-rate constant from $\frac{\kappa-1}{\kappa+1}$ to $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ |
| AdaGrad | Diagonal preconditioning — approximates dividing by Hessian diagonal, at the cost of a monotonically vanishing step size |
| RMSProp | Same diagonal preconditioning, but with a *local* (EMA) estimate instead of an all-time sum, so it can't decay to zero |
| Adam | Combines direction-smoothing (short EMA) and scale-adaptation (long EMA), with bias correction for both | 
| AdamW | Removes an unintended coupling between the scale-adaptation term and the regularization term |

## Suggested next step
In the notebook's toy non-convex surface, try artificially increasing its condition number $\kappa$ (e.g. stretch one axis of the loss bowl) and compare how many iterations plain GD, Momentum, and Adam each need to reach a fixed loss threshold — you should see the gap between them widen as $\kappa$ grows, exactly as the rate formulas in Section 1 predict.