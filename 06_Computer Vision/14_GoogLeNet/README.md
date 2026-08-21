# 14. GoogLeNet

## Learning Objectives

- Explain the Inception module: running multiple filter sizes in parallel and concatenating the results
- Understand how 1x1 convolutions reduce computational cost without losing spatial information
- Implement an Inception module and a simplified GoogLeNet in PyTorch

## The Problem

VGG (Lesson 13) showed that depth helps, but simply stacking more 3×3 layers scales parameters and compute roughly linearly with depth, and VGG-16's 138 million parameters were already large and slow. GoogLeNet (Szegedy et al., 2014 — the "Inception" architecture, named partly as a nod to the meme "we need to go deeper") won the 2014 ImageNet competition with a completely different strategy: instead of going deeper with a fixed filter size, go *wider* at each layer, using several filter sizes in parallel, while using a specific trick to keep the computational cost from exploding.

## The Concept

### The core idea: why choose one filter size?

Every architecture so far picks one filter size per layer — LeNet's 5×5, VGG's uniform 3×3. But different objects in an image appear at different scales: a small filter is well-suited to a small, local feature; a large filter better captures a larger-scale pattern. GoogLeNet's answer: don't choose — run several filter sizes on the same input, in parallel, within a single layer (an **Inception module**), and let the network combine their outputs.

```
Inception module (naive version):

                    Input feature map
                          |
        -----------------+-----------------+-------------------
        |                |                 |                  |
   1x1 conv          3x3 conv          5x5 conv          3x3 max pool
        |                |                 |                  |
        -----------------+-----------------+-------------------
                          |
              Concatenate along channel dimension
                          |
                       Output
```

Each branch processes the same input independently, and their outputs (all kept at the same spatial size via appropriate padding) are stacked along the channel dimension — this is why every branch must produce the *same* spatial output size, even though they use different filter sizes; padding is chosen per-branch specifically to make that true.

### The problem with the naive version: computational cost

Running a 5×5 convolution directly on a feature map with many channels (say, 256) is expensive: `5 * 5 * 256 * C_out` multiply-adds per output position. Stack several Inception modules and this becomes prohibitive.

### The fix: 1x1 convolutions as a "bottleneck"

A 1×1 convolution (Lesson 02's convolution operation, with `f=1`) doesn't look at any spatial neighborhood at all — it operates purely across channels, taking a weighted combination of all input channels at each individual spatial position. Used *before* the expensive 3×3 and 5×5 convolutions, a 1×1 conv can reduce the number of channels first, cutting the cost of the larger convolution that follows dramatically:

```
Inception module with 1x1 bottleneck (the actual GoogLeNet design):

Input (256 channels)
    |
    +-- 1x1 conv (64 out) --------------------------→ 1x1 branch output
    |
    +-- 1x1 conv (96 out) --→ 3x3 conv (128 out) ---→ 3x3 branch output
    |
    +-- 1x1 conv (16 out) --→ 5x5 conv (32 out)  ---→ 5x5 branch output
    |
    +-- 3x3 max pool -------→ 1x1 conv (32 out)  ---→ pool branch output
    |
    Concatenate: 64 + 128 + 32 + 32 = 256 output channels
```

Cost comparison for the 5×5 branch specifically, going from 256 input channels to 32 output channels:

```
Without 1x1 bottleneck:
  5 * 5 * 256 * 32 = 204,800 params (and proportional compute)

With 1x1 bottleneck (256 -> 16 channels first, then 5x5 conv 16 -> 32):
  1x1 conv:  1 * 1 * 256 * 16 = 4,096
  5x5 conv:  5 * 5 * 16  * 32 = 12,800
  Total: 16,896 params

Roughly 12x fewer parameters for the same 5x5 branch.
```

This is the specific mechanism that made "run several filter sizes in parallel" computationally practical rather than prohibitively expensive.

### GoogLeNet's overall structure

GoogLeNet stacks 9 Inception modules, interspersed with occasional max pooling for downsampling, and — notably — replaces the large FC layers that dominated AlexNet's and VGG's parameter counts with **global average pooling** (Lesson 05) right before a small final FC layer:

```
Input → Conv/Pool stem → [Inception module] x 9 (with occasional pooling) → Global Avg Pool → FC(1000)
```

This single change (global average pooling instead of a large flatten + FC) is a major reason GoogLeNet has only about 5 million parameters — roughly 12x fewer than AlexNet and 27x fewer than VGG-16 — while matching or exceeding their accuracy on ImageNet.

### Auxiliary classifiers (a training aid, not used at inference)

The original GoogLeNet added two extra, smaller classifier heads attached partway through the network during training, each contributing their own loss term (weighted less than the main output's loss) to the total training objective. The intuition: injecting gradient signal at intermediate depths helps combat vanishing gradients (Lesson 06) in a 22-layer network, which was unusually deep for 2014. These auxiliary heads are discarded at inference time — only the main output is used for actual predictions. This idea (extra supervision partway through a deep network) foreshadows the more general problem of training very deep networks that ResNet (Lesson 15) addresses more directly with skip connections.

See `code/googlenet_demo.py` for an Inception module implementation, a cost comparison (with vs without the 1×1 bottleneck) matching the numbers above, and a simplified GoogLeNet-style network built by stacking Inception modules.

## Exercises

1. Implement an Inception module with the four branches shown above and confirm all four branch outputs have the same spatial size before concatenation, for a given input size.
2. Reproduce the parameter-count comparison for the 5×5 branch (with vs without the 1×1 bottleneck) and confirm the roughly 12x reduction.
3. Stack 3 Inception modules with a max pool between the 2nd and 3rd, and trace the output shape and channel count through the whole stack.
4. Compare the parameter count of your simplified GoogLeNet-style network (using global average pooling) against a version that instead flattens and uses a large FC layer, similar to VGG's classifier head.

## Key Terms

| Term | What it actually means |
|---|---|
| GoogLeNet (Inception) | A 2014 CNN architecture (Szegedy et al.) that won ImageNet using parallel multi-scale filters (Inception modules) instead of pure depth |
| Inception module | A layer that applies several filter sizes (and pooling) in parallel to the same input and concatenates their outputs along the channel dimension |
| 1x1 convolution | A convolution with a 1x1 filter, which combines information across channels at each spatial position without looking at any neighborhood, often used to reduce channel count cheaply |
| Bottleneck layer | A 1x1 convolution used to reduce channel count before an expensive larger convolution, cutting computational cost |
| Auxiliary classifier | An extra classifier head attached at an intermediate depth during training only, used to inject additional gradient signal into a very deep network |
