# 09. Backpropagation in CNN

## Learning Objectives

- Explain how the chain rule extends from fully connected layers to convolutional and pooling layers
- Derive and implement the gradient of a convolutional layer with respect to its filters and its input
- Implement backpropagation through max pooling using the argmax positions recorded in Lesson 08

## The Problem

Forward propagation (Lesson 08) computes a prediction and a loss. To improve the weights, you need to know how much each weight contributed to that loss — the gradient of the loss with respect to every weight in the network. For a fully connected network this is standard backpropagation via the chain rule. Convolutional and pooling layers need their own versions of that chain rule, because their forward operations (sliding filters, taking a max over a window) aren't the same as a plain matrix multiply.

## The Concept

### The chain rule, layer by layer, backward

Backpropagation computes gradients starting from the loss and working backward through the network, at each layer using the chain rule:

```
dLoss/d(layer input) = dLoss/d(layer output) * d(layer output)/d(layer input)
```

Every layer needs to define two things: how its output depends on its input (forward propagation, Lesson 08), and — for backprop — how a gradient with respect to its output translates into a gradient with respect to its input (and, for layers with weights, with respect to those weights too).

```
Loss
  ↑  dLoss/dz2         (gradient of loss w.r.t. FC output — softmax + cross-entropy gives a clean form)
FC layer
  ↑  dLoss/df1         (gradient w.r.t. FC input, i.e. the flattened pooled features)
Flatten (reshape backward = just reshape the gradient back to pooled feature map shape)
  ↑  dLoss/dp1
MaxPool
  ↑  dLoss/da1         (gradient w.r.t. the ReLU output, before pooling)
ReLU
  ↑  dLoss/dz1         (gradient w.r.t. the conv output, before ReLU)
Conv layer
  ↑  dLoss/dx          (gradient w.r.t. the original input — not needed unless there's an earlier layer)
```

Each layer's backward step consumes the gradient flowing in from the layer after it, and produces the gradient to hand to the layer before it — plus, for layers with weights (conv, FC), the gradient with respect to *those weights*, which is what the optimizer actually uses to update them (Lesson 10).

### Backprop through ReLU

ReLU's forward pass is `max(0, z)`. Its gradient is 1 where the input was positive, 0 where it was negative or zero (Lesson 06):

```
dLoss/dz = dLoss/da * (1 if z > 0 else 0)
```

This is why the forward pass needs to be cached (Lesson 08): backprop through ReLU needs to know the *sign* of the original pre-activation `z`, which isn't recoverable from the output `a` alone once it's been zeroed.

### Backprop through max pooling

Max pooling has no weights, so there's nothing to update — but a gradient still needs to flow backward to the layer before it. Since the forward pass only used the *maximum* value in each window, only that position should receive gradient; every other position in the window gets zero, because changing them didn't affect the output at all.

```
Forward:  window = [1, 5, 3, 2]  ->  max = 5  (at position 1)

Backward: incoming gradient for this output = g
          gradient routed to window = [0, g, 0, 0]
                                          ^
                                  only the argmax position gets the gradient
```

This is exactly why Lesson 08's `max_pool_with_argmax` recorded the argmax positions during the forward pass — backprop through max pooling is a lookup into that cache, not a new computation.

### Backprop through a convolutional layer

This is the least obvious one. A conv layer's forward pass is a sliding-window dot product (Lesson 02). Its backward pass has two parts:

**Gradient with respect to the filter weights** — for each filter, sum up, across every position the filter was applied, the outer product of the incoming gradient at that output position and the input patch that produced it:

```
dLoss/dW[filter] = sum over all output positions (i,j):
    dLoss/dz[i,j] * input_patch[i,j]
```

Intuitively: a filter weight's gradient asks "across every place this filter looked at the image, how much would nudging this weight have changed the loss, weighted by how much the loss cared about that particular output position?"

**Gradient with respect to the input** — this turns out to be equivalent to convolving the incoming gradient with the *flipped* filter (this is where the "true," flipping definition of convolution from Lesson 02 reappears naturally, even though the forward pass didn't flip anything):

```
dLoss/dx = full_convolve(dLoss/dz, flip(filter))
```

Every deep learning framework computes both of these automatically via **automatic differentiation** — you never derive them by hand in practice — but understanding that they follow directly from the same chain rule used everywhere else demystifies why `.backward()` "just works" for a conv layer the same way it does for a linear layer.

### Automatic differentiation

Frameworks like PyTorch build a computation graph during the forward pass (this is exactly the "cache" from Lesson 08, generalized) and then walk it backward, applying each operation's known backward rule (the ones just derived above, for every layer type) via `.backward()`. This is why hand-deriving backprop matters for understanding, even though you'll essentially never write it by hand in real projects — Lesson 10 onward relies entirely on `loss.backward()`.

See `code/backprop_demo.py` for a from-scratch backward pass through Conv → ReLU → MaxPool → FC → Softmax, with every computed gradient verified against PyTorch's `autograd`.

## Exercises

1. By hand, compute the max-pooling backward pass for the 4×4 example from Lesson 05, given an incoming gradient of 1.0 at every pooled output position. Confirm most input positions receive zero gradient.
2. Implement `relu_backward` and verify it against `torch.autograd` by comparing gradients on a small random input.
3. Implement the convolutional filter gradient formula above for a single-channel, single-filter case, and verify it against PyTorch's autograd-computed `conv.weight.grad` after a `.backward()` call on the same forward computation.
4. Extend the exercise above to also compute the gradient with respect to the input (`dLoss/dx`) using the flipped-filter convolution, and verify against `torch.autograd`.

## Key Terms

| Term | What it actually means |
|---|---|
| Backpropagation | Computing the gradient of the loss with respect to every weight in a network by applying the chain rule backward through each layer |
| Chain rule | The calculus rule for computing the derivative of a composition of functions, applied layer by layer during backpropagation |
| Argmax routing | In max pooling's backward pass, sending the full incoming gradient to only the position that produced the maximum in the forward pass, and zero elsewhere |
| Automatic differentiation (autograd) | A framework's mechanism for automatically computing exact gradients by tracking operations during the forward pass and applying known backward rules |
| Computation graph | The recorded structure of operations from a forward pass, used by autograd to determine how to propagate gradients backward |
