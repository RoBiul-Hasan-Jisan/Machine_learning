# 05 · Neural Network Initialization 

A deeper, more mathematical companion to the initialization lesson. The organizing idea throughout: initialization is a variance-propagation problem, and every scheme here is a solution to the equation "how big should each weight be so that variance neither shrinks nor grows layer to layer."

---

## 0. Setup — Variance as the Quantity to Control

Consider a single linear layer with $n_{\text{in}}$ inputs:

$$
z = \sum_{i=1}^{n_{\text{in}}} w_i x_i
$$

Assume weights $w_i$ are i.i.d. with mean 0 and variance $\text{Var}(w)$, and inputs $x_i$ are i.i.d. with mean 0 and variance $\text{Var}(x)$, independent of the weights. Then:

$$
\text{Var}(z) = \sum_{i=1}^{n_{\text{in}}} \text{Var}(w_i x_i) = n_{\text{in}} \cdot \text{Var}(w)\cdot \text{Var}(x)
$$

This one identity is the entire mathematical basis for every initialization scheme below: it says the variance of a layer's *output* is the variance of its *input*, multiplied by $n_{\text{in}} \cdot \text{Var}(w)$. If that multiplier isn't $\approx 1$, variance compounds geometrically with depth:

$$
\text{Var}(z^{(L)}) \approx \text{Var}(x^{(0)}) \cdot \prod_{\ell=1}^{L} \left(n_{\text{in}}^{(\ell)} \cdot \text{Var}(w^{(\ell)})\right)
$$

If the per-layer multiplier is $c$, this is $\text{Var}(x^{(0)}) \cdot c^L$ — **exponential in depth**. Any $c \ne 1$ blows up or collapses for large $L$; this is the precise mathematical origin of vanishing/exploding activations (and, via the chain rule, gradients).

**The design principle for every scheme in this lesson:** choose $\text{Var}(w)$ so that $n_{\text{in}}\cdot\text{Var}(w) \approx 1$, i.e.

$$
\text{Var}(w) \approx \frac{1}{n_{\text{in}}}
$$

Everything below is a refinement of this one equation for a specific activation function and for both the forward *and* backward pass.

---

## 1. Why Initialization Matters — the Backward-Pass Mirror

The forward-pass argument above has an exact mirror for gradients. By the chain rule, the gradient flowing backward into layer $\ell$'s pre-activation is (roughly, ignoring the activation derivative for a moment):

$$
\delta^{(\ell)} = \left(W^{(\ell+1)}\right)^\top \delta^{(\ell+1)}
$$

which is structurally the *same* kind of weighted sum as the forward pass, but summing over $n_{\text{out}}$ (the fan-out of layer $\ell$, i.e. how many downstream units layer $\ell$'s output feeds into) instead of $n_{\text{in}}$:

$$
\text{Var}(\delta^{(\ell)}) = n_{\text{out}} \cdot \text{Var}(w) \cdot \text{Var}(\delta^{(\ell+1)})
$$

**This is why fan-in alone isn't quite enough:** stabilizing the forward pass wants $\text{Var}(w) = 1/n_{\text{in}}$, but stabilizing the backward pass wants $\text{Var}(w) = 1/n_{\text{out}}$. When $n_{\text{in}} \ne n_{\text{out}}$ (true for almost every real layer), you cannot satisfy both exactly — which is precisely the tension Xavier initialization resolves by compromise (Section 3), and which is why some schemes are described as "fan-in mode" vs. "fan-out mode" in code (e.g. PyTorch's `kaiming_init(mode='fan_in')` vs `mode='fan_out'`).

---

## 2. Random Initialization — Why Symmetry Breaking Is a Hard Requirement

**All-zero initialization, proven formally:** if $W^{(\ell)} = 0$ (or any constant) for all weights in a layer, then for any two units $j, k$ in that layer, $z_j = z_k$ for every input (since they compute the same weighted sum of the same inputs with identical weights). By induction, their gradients $\partial L/\partial w_j = \partial L/\partial w_k$ are also identical at every step. So the update $w_j \leftarrow w_j - \eta \partial L/\partial w_j$ keeps $w_j = w_k$ for all time — the symmetry is never broken, no matter how many steps are taken. This is why *some* source of randomness across units is not optional.

**Why the scale of that randomness still matters, quantitatively:** using $\text{Var}(w) = 1/n_{\text{in}} \cdot k$ for some fixed constant $k$ unrelated to $n_{\text{in}}$ (e.g. a fixed $\sigma=1$ or $\sigma=0.01$ regardless of layer width) means the per-layer multiplier $n_{\text{in}} \cdot \text{Var}(w) = k$ is constant across layers *only if* every layer has the same width — but even then, unless $k\approx 1$, Section 0's exponential-in-depth blowup/collapse still applies. A fixed small $\sigma$ (e.g. 0.01) makes $k \ll 1$ for any reasonably wide layer, guaranteeing vanishing; this is exactly why "small random values" fails in deep nets even though it avoids the exact-zero symmetry problem.

---

## 3. Xavier/Glorot — Deriving the Formula

**Assumptions used in the derivation:** activation is roughly linear near 0 (true for Tanh, and true for Sigmoid after re-centering), weights and inputs are independent with zero mean.

**Forward pass wants:** $\text{Var}(w) = 1/n_{\text{in}}$
**Backward pass wants:** $\text{Var}(w) = 1/n_{\text{out}}$

Glorot & Bengio's compromise is the harmonic-mean-like average:

$$
\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

This satisfies neither bound exactly, but bounds the "damage" on both passes symmetrically — when $n_{\text{in}} \approx n_{\text{out}}$ (common in the middle of a network), it's close to *both* $1/n_{\text{in}}$ and $1/n_{\text{out}}$ simultaneously.

**Normal vs. uniform variant:** for a uniform distribution $U(-a, a)$, the variance is $a^2/3$. Setting $a^2/3 = \frac{2}{n_{\text{in}}+n_{\text{out}}}$ and solving for $a$:

$$
a = \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}
$$

— which is exactly the bound given in the uniform variant. The normal and uniform variants are just two different distributional choices that both hit the same target variance $\frac{2}{n_{\text{in}}+n_{\text{out}}}$.

**Why "near-linear" matters for the derivation to hold:** the variance identity in Section 0 assumed $z = \sum w_i x_i$ propagates directly to the next layer's *input*. But the actual next input is $\sigma(z)$ for activation $\sigma$, not $z$ itself. If $\sigma$ is approximately linear with slope $\approx 1$ near 0 (true of Tanh; true of re-centered Sigmoid, whose slope at 0 is exactly $1/4$, which is why Xavier is sometimes scaled by an extra factor of 4 specifically for raw Sigmoid), then $\text{Var}(\sigma(z)) \approx \text{Var}(z)$, and the identity carries through layer to layer unmodified.

---

## 4. He Initialization — Where the Factor of 2 Comes From

**The problem Xavier's derivation doesn't account for:** ReLU is *not* linear around zero — it's zero for all negative inputs. This breaks the "$\text{Var}(\sigma(z))\approx \text{Var}(z)$" assumption Xavier relies on.

**Deriving the correction:** assume $z$ (the pre-activation) is symmetric around 0 (a reasonable assumption if weights are zero-mean and the previous layer's output is roughly symmetric). Then exactly half the mass of $z$'s distribution is negative, and ReLU maps all of that half to exactly 0:

$$
\mathbb{E}[\text{ReLU}(z)^2] = \mathbb{E}[z^2 \mid z>0]\cdot P(z>0) = \frac{1}{2}\mathbb{E}[z^2] = \frac{1}{2}\text{Var}(z)
$$

(using $\mathbb{E}[z]=0 \Rightarrow \text{Var}(z) = \mathbb{E}[z^2]$, and by symmetry $\mathbb{E}[z^2\mid z>0] = \mathbb{E}[z^2]$). So:

$$
\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)
$$

**This factor of $\frac{1}{2}$ is exactly what He initialization compensates for.** Redo the Section 0 forward-variance identity, but now propagating through a ReLU layer:

$$
\text{Var}(z^{(\ell+1)}) = n_{\text{in}} \cdot \text{Var}(w^{(\ell+1)}) \cdot \text{Var}(\text{ReLU}(z^{(\ell)})) = n_{\text{in}} \cdot \text{Var}(w^{(\ell+1)}) \cdot \frac{1}{2}\text{Var}(z^{(\ell)})
$$

For the multiplier $n_{\text{in}}\cdot\text{Var}(w)\cdot\frac{1}{2}$ to equal 1 (the stability condition from Section 0):

$$
\text{Var}(w) = \frac{2}{n_{\text{in}}}
$$

— exactly He initialization's formula. The "2" is not an empirical tuning constant; it's derived directly from the fact that ReLU discards exactly half the variance of a zero-centered, symmetric input.

---

## 5. Vanishing Gradients — the Exact Chain-Rule Product

For a network with $L$ layers, the gradient of the loss with respect to layer 1's pre-activation is, by repeated application of the chain rule:

$$
\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial z^{(L)}} \cdot \prod_{\ell=2}^{L} \left(\left(W^{(\ell)}\right)^\top \text{diag}\!\left(\sigma'(z^{(\ell-1)})\right)\right)
$$

Taking a rough scalar/variance view (ignoring matrix structure for intuition), the *magnitude* of this product depends on the product of $\|W^{(\ell)}\| \cdot |\sigma'(z^{(\ell-1)})|$ across all $L-1$ terms. Two separate quantities can each drag this toward zero:

1. **Weight magnitude:** if $\text{Var}(w) \ll 1/n_{\text{in}}$ (the Section 0 imbalance), the product of $\|W^{(\ell)}\|$ terms shrinks geometrically regardless of activation.
2. **Activation derivative:** Sigmoid's derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ has a **maximum value of exactly $0.25$** (at $z=0$) and decays toward 0 as $|z|$ grows (the "saturating tails"). Even with *perfect* weight initialization, if activations drift into the saturating region, every $\sigma'(z^{(\ell)})$ factor is $\ll 1$, and the product over many layers vanishes purely from this term — this is why Sigmoid/Tanh remain vanishing-gradient-prone even with Xavier init, and why ReLU (whose derivative is exactly 1 for all positive inputs, not just near 0) is structurally more resistant to this specific failure mode.

**Why skip connections fix this independently of initialization:** a residual block computes $z^{(\ell)} = f(z^{(\ell-1)}) + z^{(\ell-1)}$ (identity + a learned function). Differentiating:

$$
\frac{\partial z^{(\ell)}}{\partial z^{(\ell-1)}} = \frac{\partial f}{\partial z^{(\ell-1)}} + I
$$

The added identity matrix $I$ means the gradient has a direct, un-multiplied path backward through every layer (the "+1" term), so the product across $L$ layers can never fully collapse to zero even if $\partial f/\partial z^{(\ell-1)}$ is small at some layers — it provides a lower bound on gradient flow that initialization alone cannot.

---

## 6. Exploding Gradients — the Mirror-Image Failure

Same product as Section 5, but now each factor has magnitude $>1$ — e.g. if $\text{Var}(w) \gg 1/n_{\text{in}}$, or if activations are in a regime where $|\sigma'(z)|>1$ (impossible for Sigmoid, but *possible* for some other activation shapes and definitely possible for unnormalized linear/ReLU chains, where there's no ceiling on $\sigma'$ at all — ReLU's derivative is exactly 1 or 0, never $>1$, but the *weight matrix norms themselves* are what typically drive explosion in ReLU nets).

**Gradient clipping, formalized:** rather than fixing the initialization (which only controls the *expected* behavior at $t=0$, before any training has occurred), clipping directly bounds the gradient's norm at every step:

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|}\right)
$$

for some threshold $c$. This leaves the gradient's *direction* unchanged (it's a positive scalar rescaling) while capping the *step size* — a deliberately blunt, always-on safeguard against explosion that doesn't rely on the network's variance staying well-behaved throughout training the way initialization alone does.

---

## Summary: One Equation, Four Consequences

Everything in this lesson traces back to a single stability condition:

$$
n_{\text{in}} \cdot \text{Var}(w) \cdot (\text{variance scaling factor of the activation}) \approx 1
$$

| Scheme | Activation's variance-scaling factor | Resulting $\text{Var}(w)$ |
|---|---|---|
| Naive fixed-$\sigma$ | Unaccounted for | Fixed, ignores $n_{\text{in}}$ → depth-dependent blowup/collapse |
| Xavier/Glorot | $\approx 1$ (near-linear Tanh/Sigmoid) | $2/(n_{\text{in}}+n_{\text{out}})$, balancing forward & backward |
| He | $\frac{1}{2}$ (ReLU discards negative half) | $2/n_{\text{in}}$, exactly compensating the factor of $\frac{1}{2}$ |

Vanishing/exploding gradients are what happens when this equation's product is respectively $<1$ or $>1$, compounded exponentially over depth (Section 0); good initialization is choosing $\text{Var}(w)$ so the product starts at $\approx 1$, while architectural tools like skip connections and training-time tools like gradient clipping guard against the same failure mode persisting or re-emerging *after* training has begun perturbing the network away from its initial, carefully-balanced state.

## Suggested next step
In the notebook's 15+ layer NumPy network, log $\text{Var}(z^{(\ell)})$ at every layer for all three initialization schemes, and check empirically whether the naive scheme's variance changes geometrically with depth (i.e. plot on a log scale — it should look linear) as predicted by the $c^L$ formula in Section 0.