# 05 · Neural Network Initialization

Before a single gradient step is taken, the values you assign to a network's weights already determine whether training will succeed, crawl, or fail outright. This is one of the most under-appreciated topics in deep learning: two networks with identical architecture, data, and optimizer can behave completely differently just because of how their weights started.

## What you'll learn
- **Why initialization matters** — the specific failure modes bad initialization causes
- **Random initialization** — the naive baseline, and why it isn't enough on its own
- **Xavier/Glorot initialization** — designed for Sigmoid/Tanh activations
- **He initialization** — designed for ReLU-family activations
- **Vanishing gradients** — when gradients shrink to ~0 as they propagate backward through many layers
- **Exploding gradients** — the opposite failure: gradients grow uncontrollably large

---

## 1. Why Initialization Matters

Two competing failure modes emerge purely from the *scale* of a network's initial weights, compounded across every layer:

- **Weights too small** → activations shrink toward zero as they pass through each layer → gradients (computed via the chain rule, which multiplies derivatives layer by layer) shrink too → **vanishing gradients** → early layers barely update, learning is painfully slow or stalls entirely.
- **Weights too large** → activations grow explosively layer by layer → gradients also grow uncontrollably → **exploding gradients** → weight updates become huge and unstable, loss oscillates wildly or becomes `NaN`.

The deeper the network, the worse both effects compound — this is precisely why naive initialization becomes catastrophic in genuinely deep networks (dozens+ of layers), even though it might look fine in a 2-layer toy example.

## 2. Random Initialization (the naive baseline)

The obvious first idea: just sample weights from a standard normal distribution, e.g. `W ~ N(0, 1)`, or a small fixed constant like `W ~ N(0, 0.01)`.

**Problems:**
- **All-zero initialization** is catastrophic in a different way: every neuron in a layer computes the *exact same* output and receives the *exact same* gradient (perfect symmetry), so they never differentiate from each other no matter how long you train — the network effectively behaves as if it had only 1 neuron per layer. This is why weights (though not necessarily biases) must be initialized with some randomness.
- **Large random values** (e.g. `N(0,1)` in a deep net) tend to cause exploding activations/gradients.
- **Very small random values** (e.g. `N(0, 0.01)`) tend to cause vanishing activations/gradients in deep networks.

The right scale of randomness depends on the number of inputs feeding into each neuron (the "fan-in") — which is exactly the insight Xavier and He initialization formalize.

## 3. Xavier/Glorot Initialization

**Introduced for:** Sigmoid/Tanh activations (Glorot & Bengio, 2010). The goal: keep the *variance* of activations (forward pass) and gradients (backward pass) roughly constant across layers, so neither vanishes nor explodes as depth increases.

```
W ~ N(0, 2/(fan_in + fan_out))              [normal variant]
W ~ Uniform(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))   [uniform variant]
```

`fan_in` = number of inputs to the layer, `fan_out` = number of outputs. Averaging fan_in and fan_out balances keeping variance stable on both the forward pass (fan_in matters most) and the backward pass (fan_out matters most).

**Why it works for Sigmoid/Tanh specifically:** these activations are roughly linear near zero, so keeping the *pre-activation* variance close to 1 keeps most values in that near-linear (non-saturating) region, avoiding the vanishing gradients that come from saturating far out on the Sigmoid/Tanh curve's flat tails.

## 4. He Initialization

**Introduced for:** ReLU (and Leaky ReLU) activations (He et al., 2015). Xavier assumes a roughly symmetric, near-linear activation around zero — but ReLU zeroes out *all* negative inputs, effectively halving the variance that passes through compared to what Xavier assumes. He initialization compensates for that:

```
W ~ N(0, 2/fan_in)
```

The factor of `2` (instead of Xavier's factor from `fan_in+fan_out`) precisely accounts for ReLU discarding, on average, half of its inputs (everything negative), keeping the variance of activations stable through many ReLU layers.

**Rule of thumb:** use He initialization for ReLU/Leaky ReLU hidden layers (the deep learning default today), and Xavier/Glorot for Sigmoid/Tanh layers (or the output layer when it uses Sigmoid/Softmax).

## 5. Vanishing Gradients

Recall the chain rule from lesson 01: `∂L/∂W⁽¹⁾` is a *product* of many terms, one per layer between the loss and that early weight. If each of those terms has magnitude `< 1` (common with Sigmoid/Tanh, whose derivative maxes out at 0.25 for Sigmoid and 1.0 for Tanh but is usually much smaller away from zero), the product shrinks exponentially with depth:

```
∂L/∂W⁽¹⁾ ≈ (small #) × (small #) × ... × (small #)   [one factor per layer]  →  ≈ 0
```

**Symptom:** early layers' weights barely change during training, while later layers may still be learning — the network effectively can't learn deep, hierarchical features because the signal that should teach the early layers never meaningfully arrives.

**Mitigations:** He/Xavier initialization (this lesson), ReLU-family activations instead of Sigmoid/Tanh in hidden layers (lesson 01), Batch Normalization (lesson 04), and architectural solutions like residual/skip connections (used in ResNets and Transformers) that give gradients a more direct path backward.

## 6. Exploding Gradients

The opposite failure: if each layer's chain-rule term has magnitude `> 1`, the product grows exponentially with depth instead of shrinking:

```
∂L/∂W⁽¹⁾ ≈ (large #) × (large #) × ... × (large #)  →  ≈ ∞ (or NaN)
```

**Symptom:** loss suddenly spikes to a huge value or becomes `NaN`/`Inf`; weight updates become enormous and destructive.

**Mitigations:** proper initialization (this lesson), **gradient clipping** (cap the gradient's norm at a maximum value before applying the update — very common in RNN/Transformer training), lower learning rates, and Batch Normalization.

## Run the code
[`05-neural-network-initialization.ipynb`] — builds a genuinely deep (15+ layer) NumPy network with three different initialization schemes (all-zero/naive-large, Xavier, He) and tracks activation/gradient magnitude by layer depth, visually demonstrating vanishing and exploding gradients — then reproduces the comparison using `torch.nn.init`.
