# 01 · Neural Network Fundamentals

Everything in deep learning is built on a handful of ideas from this lesson: a single artificial neuron, how neurons connect into layers, how signal flows forward, how "wrongness" is measured, and how that wrongness flows backward to update every weight. Master this lesson and every later one is a variation on it.

## What you'll learn
- **Perceptron** — the original artificial neuron (1958, Rosenblatt)
- **Neural Network architecture** — layers, weights, biases, how a "network" is just perceptrons stacked and connected
- **Forward propagation** — how input becomes output, layer by layer
- **Activation functions** — Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax
- **Loss / Cost functions** — MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Backpropagation** — how the network learns *which* weights to blame for the error
- **Chain rule** — the calculus that makes backpropagation possible
- **Gradient Descent** — how weights actually get updated, step by step

---

## 1. The Perceptron

The perceptron is a single neuron that takes weighted inputs, sums them, adds a bias, and passes the result through an activation function:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w·x + b
output = activation(z)
```

- `x` — inputs
- `w` — weights (learned; how much each input matters)
- `b` — bias (learned; shifts the decision boundary)
- `activation` — a nonlinear function (see below)

A single perceptron with a step activation can only separate data that is **linearly separable** (think a single straight line/plane). This is the perceptron's famous limitation (Minsky & Papert, 1969) — it cannot learn XOR. This limitation is exactly why we stack perceptrons into networks.

## 2. Neural Network Architecture

A neural network is perceptrons arranged in **layers**:

- **Input layer** — one node per input feature (no computation, just passes values in)
- **Hidden layer(s)** — every node computes `z = w·x + b` then applies an activation function; each layer's output becomes the next layer's input
- **Output layer** — produces the final prediction; its activation depends on the task (Sigmoid for binary classification, Softmax for multi-class, linear/none for regression)

Stacking layers with **nonlinear** activations is what lets a network approximate arbitrarily complex, non-linear decision boundaries (this is the intuition behind the *Universal Approximation Theorem*). Without nonlinear activations, stacking layers is mathematically pointless — any stack of purely linear layers collapses into one equivalent linear layer.

**Depth vs width:** a "deep" network has many layers; a "wide" network has many neurons per layer. Depth tends to be more parameter-efficient for learning hierarchical features (edges → shapes → objects, in vision), which is why "deep" learning is named for depth, not width.

## 3. Forward Propagation

Forward propagation is simply applying the perceptron equation layer by layer until you reach the output:

```
z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾        a⁽¹⁾ = activation(z⁽¹⁾)
z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾      a⁽²⁾ = activation(z⁽²⁾)
...
ŷ = a⁽ᴸ⁾   (output of the final, L-th layer)
```

`W⁽ˡ⁾` is now a **matrix** (one row per neuron in layer `l`, one column per input from layer `l-1`), because each layer has many neurons, not one. This is why deep learning is fundamentally matrix multiplication at scale — and why GPUs (built for parallel matrix math) accelerated the field so much.

## 4. Activation Functions

Without activation functions, a neural network is just linear regression no matter how many layers it has. Activations inject the nonlinearity that lets networks model complex functions.

| Function | Formula | Range | Typical use | Key issue |
|---|---|---|---|---|
| **Sigmoid** | `σ(z) = 1/(1+e⁻ᶻ)` | (0, 1) | Binary output layer | Saturates → vanishing gradients; not zero-centered |
| **Tanh** | `tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | (-1, 1) | Hidden layers (older nets) | Still saturates at extremes, though zero-centered (better than sigmoid) |
| **ReLU** | `max(0, z)` | [0, ∞) | Default for hidden layers | "Dying ReLU" — neurons stuck outputting 0 forever if z always < 0 |
| **Leaky ReLU** | `z if z>0 else αz` (small α, e.g. 0.01) | (-∞, ∞) | Fixes dying ReLU | Extra hyperparameter α; not always better in practice |
| **Softmax** | `eᶻᵢ / Σⱼeᶻʲ` | (0,1), sums to 1 | Multi-class output layer | Only makes sense on the *output* layer, turns logits into a probability distribution |

**Why ReLU won:** it's computationally trivial (just a max), doesn't saturate for positive inputs (gradient is a constant 1, so gradients don't vanish for active neurons), and empirically trains faster than Sigmoid/Tanh on deep networks.

## 5. Loss / Cost Functions

The loss function measures how wrong a prediction is; the network's entire job is to minimize it.

- **Mean Squared Error (MSE)** — regression: `L = (1/n)Σ(y - ŷ)²`. Penalizes large errors heavily (squared), sensitive to outliers.
- **Binary Cross-Entropy (BCE)** — binary classification: `L = -(1/n)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]`. Pairs naturally with Sigmoid output; heavily penalizes confident wrong predictions.
- **Categorical Cross-Entropy (CCE)** — multi-class classification: `L = -(1/n)ΣΣ yᵢⱼ·log(ŷᵢⱼ)`. Pairs naturally with Softmax output.

**Why not MSE for classification?** MSE with Sigmoid produces a non-convex loss landscape with much weaker gradients when predictions are very wrong — it slows learning. Cross-entropy's gradient is proportional to `(ŷ - y)`, giving a strong, well-behaved learning signal exactly when the model is most wrong.

## 6. Backpropagation & the Chain Rule

Backpropagation answers: *"given the loss, how much should each individual weight in the network change?"*

It works by applying the **chain rule** of calculus, layer by layer, from the output back to the input:

```
∂L/∂W⁽ˡ⁾ = ∂L/∂a⁽ᴸ⁾ · ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ · ∂z⁽ᴸ⁾/∂a⁽ᴸ⁻¹⁾ · ... · ∂a⁽ˡ⁾/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
```

In practice this is computed efficiently in two passes:
1. **Forward pass** — compute and cache every layer's `z` and `a`
2. **Backward pass** — starting from the loss, propagate the gradient backward, reusing the cached values, one layer at a time

Each layer only needs to know: (a) the gradient flowing in from the layer after it, and (b) its own local derivative. This is why backprop is efficient — it's `O(number of weights)`, not exponential in depth.

## 7. Gradient Descent

Once you have `∂L/∂W` for every weight, gradient descent updates each weight by stepping *against* the gradient (downhill on the loss surface):

```
W := W - η · ∂L/∂W
```

- `η` (eta) is the **learning rate** — how big a step to take. Too large → overshoot/diverge. Too small → painfully slow convergence.
- This is repeated for many iterations until the loss stops meaningfully decreasing.

Gradient descent doesn't guarantee finding the *global* minimum (the loss surface of a deep network is highly non-convex), but empirically, for large over-parameterized networks, most local minima found this way generalize surprisingly well.

## Run the code
[`01-neural-network-fundamentals.ipynb`] — builds a perceptron and a small feedforward network **twice**: first entirely from scratch with NumPy (so you implement forward prop, every activation, every loss, and manual backprop with the chain rule yourself), then again with PyTorch (so you see how `autograd` and `nn.Module` automate exactly what you just did by hand.

## Next
[`02-training-neural-networks`]— now that you can compute one gradient step, how do you actually train a network over an entire dataset?
