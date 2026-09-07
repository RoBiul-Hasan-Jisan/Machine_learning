# 01 · Neural Network Fundamentals 

Everything in deep learning is built on a handful of ideas from this lesson: a single artificial neuron, how neurons connect into layers, how signal flows forward, how "wrongness" is measured, and how that wrongness flows backward to update every weight. Master this lesson and every later one is a variation on it.

## What you'll learn
- **Perceptron** — the original artificial neuron (1958, Rosenblatt), its learning rule, and why it fails on XOR
- **Neural Network architecture** — layers, weights, biases, the Universal Approximation Theorem
- **Forward propagation** — a full worked numeric example, not just the equations
- **Activation functions** — Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax, and *why* each one behaves the way it does mathematically
- **Loss / Cost functions** — MSE, BCE, CCE, and why the choice of loss interacts with the choice of activation
- **Backpropagation** — a full worked numeric example of the chain rule flowing backward through two layers
- **Gradient Descent** — batch vs. stochastic vs. mini-batch, and a first look at momentum/Adam
- **Weight initialization & vanishing/exploding gradients** — why they matter and how they connect to everything above

---

## 1. The Perceptron

The perceptron is a single neuron that takes weighted inputs, sums them, adds a bias, and passes the result through an activation function:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w·x + b
output = activation(z)
```

- `x` — inputs
- `w` — weights (learned; how much each input matters)
- `b` — bias (learned; shifts the decision boundary independently of the inputs)
- `activation` — a nonlinear function (see §4)

### 1.1 Geometric intuition

`w·x + b = 0` defines a hyperplane. In 2D that's just a line: `w₁x₁ + w₂x₂ + b = 0`. Everything on one side of that line gets classified as one class, everything on the other side as the other class. The weights `w` control the line's *orientation* (its normal vector points in the direction of steepest increase of `z`); the bias `b` controls how far the line is shifted from the origin. Training a perceptron is literally the process of rotating and shifting this line until it separates the two classes.

### 1.2 The original learning rule

Rosenblatt's perceptron used a very simple update rule, applied one training example at a time:

```
w := w + η(y - ŷ)x
b := b + η(y - ŷ)
```

If the perceptron predicts correctly, `y - ŷ = 0` and nothing changes. If it predicts wrong, the weights are nudged in the direction that would have made the correct answer more likely. This is a primitive ancestor of gradient descent (§7) — it moves weights based on the error, just without calculus behind it (the step activation isn't differentiable, so there's no gradient to take).

**Perceptron Convergence Theorem:** if the data is linearly separable, this rule is *guaranteed* to find a separating hyperplane in a finite number of steps. If the data is *not* linearly separable, it will never converge — it will oscillate forever.

### 1.3 Why XOR broke everything (1969)

XOR truth table:

| x₁ | x₂ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Try to draw a single straight line on a 2D plot of these four points that puts the two 1s on one side and the two 0s on the other — it's provably impossible. The positive class (0,1) and (1,0) surrounds the negative class diagonally; no single line separates them. Minsky & Papert's 1969 book *Perceptrons* proved this rigorously and, because it was read (somewhat unfairly) as a proof that neural networks in general were fundamentally limited, it contributed to the first "AI winter" — funding for neural network research collapsed for over a decade.

The fix, which we now take for granted, is **stacking layers**: a hidden layer can bend the decision boundary into two lines (or a curve), which is enough to solve XOR. This single limitation is the entire historical reason multi-layer networks exist.

---

## 2. Neural Network Architecture

A neural network is perceptrons arranged in **layers**:

- **Input layer** — one node per input feature (no computation, just passes values in)
- **Hidden layer(s)** — every node computes `z = w·x + b` then applies an activation function; each layer's output becomes the next layer's input
- **Output layer** — produces the final prediction; its activation depends on the task (Sigmoid for binary classification, Softmax for multi-class, linear/none for regression)

### 2.1 Why nonlinearity is non-negotiable

Suppose every layer used a linear (identity) activation. Then:

```
a⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
a⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾ = W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾ = (W⁽²⁾W⁽¹⁾)x + (W⁽²⁾b⁽¹⁾ + b⁽²⁾)
```

That's just `W'x + b'` — a single linear layer in disguise. You could have a thousand layers and it would still only be able to represent what one layer can represent: a straight line (or hyperplane). This is why §4's activation functions aren't a minor implementation detail — they are the entire reason depth buys you anything at all.

### 2.2 The Universal Approximation Theorem, informally

The theorem (Cybenko 1989, Hornik 1991) states that a feedforward network with a single hidden layer, given *enough* neurons and a suitable nonlinear activation, can approximate any continuous function on a bounded domain to arbitrary precision. Important caveats:

- It's an *existence* proof, not a *construction* — it doesn't tell you how many neurons you need, or how to train them.
- "Enough neurons" can mean exponentially many for a shallow network to represent something a deep network represents compactly.
- This is precisely why depth is preferred in practice over pure width — see §2.3.

### 2.3 Depth vs. width

A "deep" network has many layers; a "wide" network has many neurons per layer. Depth tends to be more parameter-efficient for learning **hierarchical, compositional features** — e.g., in vision, early layers detect edges, middle layers combine edges into shapes and textures, later layers combine shapes into object parts, and the final layers combine parts into whole objects. Each layer reuses and recombines the features the previous layer already learned, which a single very-wide layer cannot do as efficiently — it has to relearn each pattern from raw input directly. This is why "deep" learning is named for depth, not width.

---

## 3. Forward Propagation

Forward propagation is applying the perceptron equation layer by layer until you reach the output:

```
z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾        a⁽¹⁾ = activation(z⁽¹⁾)
z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾      a⁽²⁾ = activation(z⁽²⁾)
...
ŷ = a⁽ᴸ⁾   (output of the final, L-th layer)
```

`W⁽ˡ⁾` is a **matrix** (one row per neuron in layer `l`, one column per input from layer `l-1`), because each layer has many neurons, not one. This is why deep learning is fundamentally matrix multiplication at scale — and why GPUs (built for parallel matrix math) accelerated the field so much.

### 3.1 Worked numeric example

Take a tiny network: 2 inputs → 2 hidden neurons (ReLU) → 1 output neuron (Sigmoid).

```
x = [1.0, 0.5]

W⁽¹⁾ = [[0.4, -0.6],     b⁽¹⁾ = [0.1, -0.2]
        [0.3,  0.8]]
```

Hidden layer pre-activation:
```
z⁽¹⁾₁ = 0.4(1.0) + (-0.6)(0.5) + 0.1 = 0.4 - 0.3 + 0.1 = 0.2
z⁽¹⁾₂ = 0.3(1.0) + 0.8(0.5) + (-0.2) = 0.3 + 0.4 - 0.2 = 0.5
```

Apply ReLU: `a⁽¹⁾ = [max(0,0.2), max(0,0.5)] = [0.2, 0.5]`

Output layer: `W⁽²⁾ = [0.7, -0.5]`, `b⁽²⁾ = 0.05`
```
z⁽²⁾ = 0.7(0.2) + (-0.5)(0.5) + 0.05 = 0.14 - 0.25 + 0.05 = -0.06
ŷ = σ(-0.06) = 1/(1+e^0.06) ≈ 0.485
```

So the network currently predicts class-1 probability ≈ 0.485. Every quantity computed here (`z⁽¹⁾, a⁽¹⁾, z⁽²⁾, ŷ`) gets **cached**, because backpropagation (§6) reuses every one of them.

---

## 4. Activation Functions

Without activation functions, a neural network is just linear regression no matter how many layers it has (§2.1). Activations inject the nonlinearity that lets networks model complex functions.

| Function | Formula | Range | Typical use | Key issue |
|---|---|---|---|---|
| **Sigmoid** | `σ(z) = 1/(1+e⁻ᶻ)` | (0, 1) | Binary output layer | Saturates → vanishing gradients; not zero-centered |
| **Tanh** | `tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | (-1, 1) | Hidden layers (older nets) | Still saturates at extremes, though zero-centered (better than sigmoid) |
| **ReLU** | `max(0, z)` | [0, ∞) | Default for hidden layers | "Dying ReLU" — neurons stuck outputting 0 forever if z always < 0 |
| **Leaky ReLU** | `z if z>0 else αz` (small α, e.g. 0.01) | (-∞, ∞) | Fixes dying ReLU | Extra hyperparameter α; not always better in practice |
| **Softmax** | `eᶻᵢ / Σⱼeᶻʲ` | (0,1), sums to 1 | Multi-class output layer | Only makes sense on the *output* layer, turns logits into a probability distribution |

### 4.1 The saturation problem, precisely

`σ'(z) = σ(z)(1 - σ(z))`, which has a maximum value of exactly 0.25 (at z=0) and shrinks toward 0 as `|z|` grows. During backprop, gradients get *multiplied* layer by layer (§6). If every layer's local gradient is at most 0.25, then after 10 layers the gradient reaching the earliest layers is bounded by roughly `0.25¹⁰ ≈ 9.5×10⁻⁷` — vanishingly small. This is the **vanishing gradient problem**, and it's the single biggest reason Sigmoid/Tanh fell out of favor for hidden layers in deep networks.

### 4.2 Why ReLU actually won

`ReLU'(z) = 1` for all `z > 0` — no shrinkage at all when the neuron is active. This means gradients pass through active ReLU neurons completely undiminished, no matter how deep the network is. Combined with being a single comparison (computationally trivial, no exponentials), this is why ReLU became the default. Its failure mode is the mirror image of Sigmoid's problem: if a neuron's `z` is negative for every training example (which can happen with a bad initialization or too-large a learning rate pushing weights into a bad region), its gradient is exactly 0 forever, and it "dies" — Leaky ReLU's small slope `α` for `z<0` exists specifically to give dead neurons a nonzero gradient to recover from.

### 4.3 Softmax intuition

Softmax takes a vector of raw scores ("logits") and turns them into a probability distribution: exponentiate every entry to make everything positive, then divide by the sum so the total is 1. Exponentiating also has an important side-effect: it exaggerates differences between logits, so a slightly-higher logit for the correct class translates into a disproportionately higher probability — this is what makes softmax's outputs "peaky" and confident rather than washed-out.

---

## 5. Loss / Cost Functions

The loss function measures how wrong a prediction is; the network's entire job is to minimize it.

- **Mean Squared Error (MSE)** — regression: `L = (1/n)Σ(y - ŷ)²`. Penalizes large errors heavily (squared), sensitive to outliers.
- **Binary Cross-Entropy (BCE)** — binary classification: `L = -(1/n)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]`. Pairs naturally with Sigmoid output; heavily penalizes confident wrong predictions.
- **Categorical Cross-Entropy (CCE)** — multi-class classification: `L = -(1/n)ΣΣ yᵢⱼ·log(ŷᵢⱼ)`. Pairs naturally with Softmax output.

### 5.1 Why not MSE for classification — the actual math

Combine MSE with a Sigmoid output: `L = (y - σ(z))²`. Its derivative with respect to `z` works out to:

```
∂L/∂z = -2(y - σ(z))·σ(z)(1 - σ(z))
```

Notice the `σ(z)(1-σ(z))` term — the same saturating derivative from §4.1. When the model is *very* wrong (e.g., `σ(z) ≈ 0` but `y = 1`), this term is close to 0, so the gradient is close to 0 exactly when you'd want the biggest correction. The model barely learns from its worst mistakes.

Now combine Binary Cross-Entropy with Sigmoid instead. The two derivatives cancel almost perfectly:

```
∂L/∂z = ŷ - y
```

This is clean, strong, and *linear* in the error — the more wrong the prediction, the bigger the gradient, with no saturating term dragging it back down. This cancellation is not a coincidence — it's the specific mathematical reason Sigmoid+BCE and Softmax+CCE are the standard pairings, rather than an arbitrary convention.

---

## 6. Backpropagation & the Chain Rule

Backpropagation answers: *"given the loss, how much should each individual weight in the network change?"*

It works by applying the **chain rule** of calculus, layer by layer, from the output back to the input:

```
∂L/∂W⁽ˡ⁾ = ∂L/∂a⁽ᴸ⁾ · ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ · ∂z⁽ᴸ⁾/∂a⁽ᴸ⁻¹⁾ · ... · ∂a⁽ˡ⁾/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
```

In practice this is computed efficiently in two passes:
1. **Forward pass** — compute and cache every layer's `z` and `a` (§3.1)
2. **Backward pass** — starting from the loss, propagate the gradient backward, reusing the cached values, one layer at a time

Each layer only needs to know: (a) the gradient flowing in from the layer after it (often called `δ`), and (b) its own local derivative. This is why backprop is efficient — it's `O(number of weights)`, not exponential in depth.

### 6.1 Worked numeric example (continuing §3.1)

Using the forward pass from §3.1: `ŷ ≈ 0.485`. Suppose the true label is `y = 1`, and we use BCE + Sigmoid, so by §5.1: `∂L/∂z⁽²⁾ = ŷ - y = 0.485 - 1 = -0.515`. Call this `δ⁽²⁾ = -0.515`.

**Gradient for the output layer's weights and bias:**
```
∂L/∂W⁽²⁾ = δ⁽²⁾ · a⁽¹⁾ = -0.515 × [0.2, 0.5] = [-0.103, -0.2575]
∂L/∂b⁽²⁾ = δ⁽²⁾ = -0.515
```

**Propagate the error back to the hidden layer.** First, how much does each hidden neuron's *output* affect the loss:
```
∂L/∂a⁽¹⁾ = δ⁽²⁾ · W⁽²⁾ = -0.515 × [0.7, -0.5] = [-0.3605, 0.2575]
```

Then multiply by ReLU's local derivative at each hidden neuron (`1` if `z⁽¹⁾ > 0`, else `0`). Both `z⁽¹⁾₁ = 0.2` and `z⁽¹⁾₂ = 0.5` were positive (§3.1), so both derivatives are 1:
```
δ⁽¹⁾ = [-0.3605, 0.2575] ⊙ [1, 1] = [-0.3605, 0.2575]
```

**Gradient for the hidden layer's weights:** each entry is `δ⁽¹⁾ᵢ × xⱼ`:
```
∂L/∂W⁽¹⁾ = [[-0.3605×1.0, -0.3605×0.5],
             [ 0.2575×1.0,  0.2575×0.5]]
          = [[-0.3605, -0.18025],
             [ 0.2575,  0.12875]]
∂L/∂b⁽¹⁾ = δ⁽¹⁾ = [-0.3605, 0.2575]
```

Every gradient the network needs — for both layers — came from exactly one forward pass and one backward pass, reusing `z⁽¹⁾, a⁽¹⁾, z⁽²⁾` computed earlier. This is the whole trick: no gradient is ever recomputed from scratch for a deeper layer.

---

## 7. Gradient Descent

Once you have `∂L/∂W` for every weight, gradient descent updates each weight by stepping *against* the gradient (downhill on the loss surface):

```
W := W - η · ∂L/∂W
```

- `η` (eta) is the **learning rate** — how big a step to take. Too large → overshoot/diverge. Too small → painfully slow convergence.

### 7.1 Batch, Stochastic, and Mini-Batch variants

| Variant | Gradient computed from | Pros | Cons |
|---|---|---|---|
| **Batch GD** | entire training set | Stable, accurate gradient direction | Slow per step; doesn't fit in memory for large datasets |
| **Stochastic GD (SGD)** | one random example | Fast updates; noise can help escape shallow local minima | Very noisy path to the minimum; unstable |
| **Mini-batch GD** | small random subset (e.g. 32–256 examples) | Balances stability and speed; maps well onto GPU parallelism | Batch size is an extra hyperparameter to tune |

In practice, virtually all modern training uses mini-batch gradient descent — "SGD" in most deep learning code actually refers to mini-batch GD.

### 7.2 A first look beyond vanilla gradient descent

Two widely-used refinements, worth knowing the names of even before you implement them:

- **Momentum** — accumulates a running average of past gradients and moves in that direction, which smooths out oscillations and speeds up progress in consistent directions (like a ball rolling downhill picking up speed).
- **Adam** — combines momentum with a *per-parameter* adaptive learning rate (based on recent gradient magnitudes), making it robust to poorly-scaled features and a common default optimizer in practice.

### 7.3 Non-convexity, in context

Gradient descent doesn't guarantee finding the *global* minimum — the loss surface of a deep network is highly non-convex, full of local minima and saddle points. Empirically, for large over-parameterized networks, most local minima found this way generalize surprisingly well, and saddle points (not bad local minima) turn out to be the more common obstacle in high dimensions — which momentum-based methods are specifically good at pushing through.

---

## 8. Weight Initialization & the Vanishing/Exploding Gradient Problem

This section ties §4.1, §6, and §7 together, and is often skipped in intro material despite being one of the most practically important ideas here.

- **Never initialize all weights to 0.** Every neuron in a layer would compute the identical `z`, get the identical gradient, and update identically forever — a phenomenon called the **symmetry problem**. The network would behave as if each layer had only one effective neuron, no matter how many you added.
- **Too-large random initial weights** push `z` values into the saturating regions of Sigmoid/Tanh (§4.1) from the very first forward pass, causing vanishing gradients before training even starts.
- **Too-small random initial weights** shrink `z` toward 0 at every layer, and by §2.1's linearity argument, the network behaves close to linear until it manages to grow its weights — slow and fragile.

Modern initialization schemes (**Xavier/Glorot** for Sigmoid/Tanh, **He initialization** for ReLU) scale the random initial weights based on the number of inputs/outputs of each layer specifically to keep the variance of activations roughly constant from layer to layer, avoiding both failure modes at the start of training. You'll see these by name in the accompanying notebook's PyTorch section — `nn.Linear` uses a variant of Kaiming (He) initialization by default.

---

## Run the code
[`01-neural-network-fundamentals.ipynb`] — builds a perceptron and a small feedforward network **twice**: first entirely from scratch with NumPy (so you implement forward prop, every activation, every loss, and manual backprop with the chain rule yourself — reproducing the exact numbers in §3.1 and §6.1), then again with PyTorch (so you see how `autograd` and `nn.Module` automate exactly what you just did by hand).