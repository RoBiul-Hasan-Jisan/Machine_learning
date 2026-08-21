# 06. Activation Functions

## Learning Objectives

- Explain why nonlinear activation functions are essential between convolutional layers
- Compare ReLU, Leaky ReLU, and GELU, and know why ReLU became the CNN default
- Recognize and address the "dying ReLU" problem

## The Problem

A convolution is a linear operation (a weighted sum). Stack several linear operations with nothing in between, and the result is still just one big linear operation — no matter how many layers you add, the network could only ever learn linear relationships between pixels and output, which is nowhere near enough to represent "this is a cat." Activation functions insert a nonlinearity after each convolution, which is what actually gives a deep stack of layers more representational power than a single layer.

## The Concept

### Why nonlinearity is non-negotiable

```
Without activation functions:
  layer2(layer1(x)) = W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
                     = W' x + b'                    <- still just ONE linear function

With activation functions:
  layer2(f(layer1(x))) where f is nonlinear
                     -> cannot be collapsed into one linear function
                     -> the network can now represent curved decision boundaries,
                        combinations of features, and much more complex patterns
```

This is true regardless of how many layers you stack — without a nonlinearity between them, a 50-layer network has no more representational power than a single linear layer.

### ReLU: the CNN default

```
ReLU(x) = max(0, x)
```

```
   output
     |
     |        /
     |       /
     |      /
     |     /
-----+----+-----------> input
     |____/
     0
```

ReLU zeroes out negative values and passes positive values through unchanged. It became the standard activation for CNNs (replacing sigmoid and tanh, which were common in earlier networks) for a few concrete reasons:

- **Cheap to compute**: just a max operation, no exponentials.
- **Reduces vanishing gradients**: sigmoid and tanh saturate (flatten out) for large positive or negative inputs, making their gradient near zero and slowing learning in deep networks. ReLU's gradient is a constant 1 for any positive input, so gradients don't shrink as they pass back through many ReLU layers on the positive side.
- **Sparse activations**: many units output exactly 0 for a given input, which empirically tends to help generalization and gives a rough analogy to biological neurons that either fire or don't.

### The dying ReLU problem

If a unit's weights update such that its input is negative for essentially every training example, its output is always 0, its gradient is always 0, and it stops updating entirely — permanently "dead." This can happen from a large gradient update (e.g. too high a learning rate) pushing weights into a regime the unit never recovers from.

```
ReLU gradient:
  x > 0:  gradient = 1     (passes gradient through normally)
  x < 0:  gradient = 0     (blocks gradient entirely - the unit can't learn to "come back")
```

### Leaky ReLU: a small fix

```
LeakyReLU(x) = x        if x > 0
             = alpha*x  if x <= 0     (alpha is small, e.g. 0.01)
```

Leaky ReLU allows a small, nonzero gradient when the input is negative, so a unit pushed into the negative region isn't permanently stuck — it can still receive a (small) gradient signal and recover. This trades a small amount of the "clean sparsity" of ReLU for robustness against dead units.

### GELU and other smooth alternatives

```
GELU(x) ≈ x * Phi(x)      where Phi is the standard normal CDF
```

GELU is a smooth curve that behaves similarly to ReLU for large |x| but transitions gradually near zero rather than having a sharp kink. It's more expensive to compute than ReLU but has become the default in transformer-based architectures (including Vision Transformers) and is used in some modern CNN designs too, generally giving small but consistent accuracy improvements. For most classic CNN work (Lessons 11-16 of this module), plain ReLU remains the standard choice — it's simpler, faster, and well understood.

### Where activation functions sit in a CNN

```
Conv → Activation → Pool → Conv → Activation → Pool → ... → Flatten → FC → Activation → FC → Output
```

Activation is applied element-wise, immediately after each convolutional (or fully connected) layer's linear output, before pooling. The final output layer typically uses a different function suited to the task — softmax for multi-class classification, sigmoid for binary classification — rather than ReLU, since the output needs to represent probabilities, not arbitrary positive activations.

See `code/activations_demo.py` for a runnable comparison of ReLU, Leaky ReLU, and GELU, including a demonstration of the dying ReLU problem and how Leaky ReLU mitigates it.

## Exercises

1. Plot ReLU, Leaky ReLU (alpha=0.01), and GELU on the same axes for `x` from -5 to 5. Note where they agree and where they diverge.
2. Construct a small example where a linear network (no activation) collapses to an equivalent single linear layer, by explicitly multiplying out the weight matrices, and confirm the outputs match.
3. Simulate a "dead" ReLU unit: initialize a single neuron's weights so its output is negative for all inputs in a small dataset, and confirm its gradient is exactly zero regardless of the target. Repeat with Leaky ReLU and observe that some gradient still flows.
4. Train the same tiny CNN from Lesson 03 with ReLU vs Leaky ReLU vs GELU and compare final training accuracy and the fraction of "dead" first-layer filters (filters whose output is zero for every training example).

## Key Terms

| Term | What it actually means |
|---|---|
| Activation function | A nonlinear function applied element-wise after a layer's linear operation, without which stacking layers has no extra representational power |
| ReLU | `max(0, x)`; the standard CNN activation function due to its simplicity and resistance to vanishing gradients on the positive side |
| Dying ReLU | The failure mode where a unit's input becomes permanently negative, making its output and gradient always zero |
| Leaky ReLU | A ReLU variant that allows a small nonzero gradient for negative inputs, mitigating dying units |
| Vanishing gradient | The problem where gradients shrink toward zero as they propagate backward through many saturating layers, slowing or stalling learning |
