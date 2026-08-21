# 08. Forward Propagation

## Learning Objectives

- Trace a complete forward pass through a CNN, from raw pixels to output scores
- Implement forward propagation for a small CNN entirely from scratch with NumPy
- Verify a from-scratch forward pass against an equivalent PyTorch model

## The Problem

Lesson 07 assembled the layers of a CNN and computed their shapes. Forward propagation is the actual computation: given a specific input image and a specific set of (already-known) weights, compute the exact numeric output, layer by layer. Understanding this concretely — as plain array operations, not framework abstractions — is what makes backpropagation (Lesson 09) make sense, since backprop is defined entirely in terms of the operations forward propagation performs.

## The Concept

### The forward pass, step by step

For a simple CNN — `Conv → ReLU → MaxPool → Flatten → FC → Softmax` — forward propagation means applying each operation in sequence, using each layer's *current* weights, and passing the output of one layer as the input to the next.

```
x                                          (raw input image)
  ↓ convolve with filters W1, add bias b1
z1 = conv(x, W1) + b1
  ↓ apply ReLU
a1 = ReLU(z1)
  ↓ max pool
p1 = maxpool(a1)
  ↓ flatten
f1 = flatten(p1)
  ↓ fully connected: multiply by weight matrix, add bias
z2 = W2 @ f1 + b2
  ↓ softmax (for classification)
y_hat = softmax(z2)                         (predicted class probabilities)
```

Every arrow above is exactly the operation covered in an earlier lesson (Lesson 02 for conv, Lesson 06 for ReLU, Lesson 05 for max pool, Lesson 07 for flatten/FC). Forward propagation is simply running them all in sequence, once, for one input.

### Softmax: turning scores into probabilities

The final FC layer produces raw scores (logits), one per class, which can be any real number. Softmax converts these into a probability distribution — nonnegative numbers that sum to 1:

```
softmax(z)_i = exp(z_i) / sum(exp(z_j) for all j)
```

In practice, subtract the max logit before exponentiating (`exp(z_i - max(z))`) for numerical stability — this doesn't change the result mathematically (it cancels out in the ratio) but avoids overflow from exponentiating large numbers.

### Why max pooling needs to remember its argmax

Forward propagation for max pooling just takes the max of each window (Lesson 05). But backpropagation (Lesson 09) will need to know *which* input position produced that max, in order to route the gradient back to the right place. A from-scratch implementation should record the argmax positions during the forward pass — a detail that's invisible when just calling `torch.nn.MaxPool2d`, since the framework handles it internally, but essential to understand for Lesson 09.

### Batches

Real training rarely runs forward propagation on a single image at a time — it processes a **batch** of images together (say, 32 or 64 at once), both for computational efficiency (better use of parallel hardware) and to get a more stable gradient estimate when the batch is later used for backpropagation. Every operation above extends naturally to a batch dimension: a conv layer processes `(batch, C_in, H, W)` and produces `(batch, C_out, H_out, W_out)`, applying the identical filters to every image in the batch independently.

### Implementing forward propagation from scratch

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)  # numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def forward_pass(x, W1, b1, W2, b2):
    z1 = conv_layer(x, W1, b1)      # from Lesson 02/07's conv_layer
    a1 = relu(z1)
    p1, argmax_positions = max_pool_with_argmax(a1)   # needed later for backprop
    f1 = p1.flatten()
    z2 = W2 @ f1 + b2
    y_hat = softmax(z2)
    return y_hat, {"z1": z1, "a1": a1, "p1": p1, "argmax": argmax_positions, "f1": f1, "z2": z2}
```

Notice the function returns not just the final prediction but also every intermediate value (a **cache**). This is deliberate: backpropagation (Lesson 09) needs these intermediate values to compute gradients, since the chain rule requires knowing the input to each layer, not just the final output.

See `code/forward_prop_demo.py` for the complete from-scratch forward pass implementation, verified numerically against an equivalent `torch.nn` model with identical weights.

## Exercises

1. Implement `softmax` from scratch and verify it produces the same output as `torch.nn.functional.softmax` on the same input logits.
2. Implement `max_pool_with_argmax`, confirming both the pooled values and the recorded argmax positions are correct for a small hand-checkable input.
3. Run a full from-scratch forward pass on a random 8×8×1 input through `Conv(4 filters, 3x3) → ReLU → MaxPool(2x2) → Flatten → FC(3 units) → Softmax`. Copy the exact same random weights into an equivalent PyTorch model and confirm the two outputs match to floating-point precision.
4. Extend your forward pass to process a batch of 5 images at once, and confirm the output shape is `(5, num_classes)`.

## Key Terms

| Term | What it actually means |
|---|---|
| Forward propagation | Computing a network's output by applying each layer's operation in sequence to an input, using the current weights |
| Logits | The raw, pre-softmax output scores of a classification network's final layer |
| Softmax | A function converting raw logits into a probability distribution (nonnegative values summing to 1) |
| Cache (in forward propagation) | The intermediate values computed during the forward pass, stored because backpropagation needs them to compute gradients |
| Batch | A group of inputs processed together through the network in one forward (and later backward) pass |
