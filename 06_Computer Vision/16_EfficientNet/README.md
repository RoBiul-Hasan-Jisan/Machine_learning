# 16. EfficientNet

## Learning Objectives

- Explain the three dimensions a CNN can be scaled along: depth, width, and resolution
- Understand compound scaling and why scaling all three dimensions together outperforms scaling any one alone
- Implement a depthwise separable convolution and explain why it's dramatically cheaper than a standard convolution

## The Problem

By 2019, the pattern for getting a more accurate CNN was usually "make it bigger" — deeper (more layers, like ResNet-152 vs ResNet-18), wider (more channels per layer), or fed higher-resolution input. But these scaling choices were typically made ad hoc, one dimension at a time, by hand, for each new model. EfficientNet (Tan & Le, 2019) asked: is there a principled way to scale a network up that uses the added compute budget optimally, instead of arbitrarily choosing to add depth *or* width *or* resolution?

## The Concept

### Three ways to scale a CNN

```
Depth scaling:       add more layers (more residual blocks per stage, Lesson 15)
Width scaling:       add more channels per layer (more filters per conv, Lesson 07)
Resolution scaling:  feed larger input images (more spatial detail per image)
```

Each dimension helps, individually, up to a point, with diminishing returns: doubling depth alone eventually stops helping much (very deep networks plateau, even with ResNet's skip connections making them trainable); doubling width alone tends to capture more fine-grained features but wastes capacity if the network isn't also deep enough to make good use of them; higher resolution alone helps the network see more detail, but without more depth/width, the network may lack the capacity to make sense of that extra detail.

### Compound scaling: scale all three together, in a fixed ratio

EfficientNet's core finding: scaling depth, width, and resolution *together*, in a carefully balanced ratio, achieves much better accuracy per unit of added compute than scaling any single dimension alone. Given a "compound coefficient" `phi` controlling how much extra compute budget you're willing to spend, EfficientNet scales all three dimensions using fixed exponents found via a small grid search on a baseline model:

```
depth:      d = alpha^phi
width:      w = beta^phi
resolution: r = gamma^phi

subject to:  alpha * beta^2 * gamma^2 ≈ 2      (keeps compute roughly doubling per unit of phi)
```

`alpha`, `beta`, `gamma` are constants found once (via search) for a baseline architecture; `phi` is then the single knob a practitioner turns to get a family of models (EfficientNet-B0 through B7) at increasing size and accuracy, all following the same balanced scaling recipe rather than an arbitrary hand-picked combination.

```
EfficientNet-B0 (baseline, phi=0)   →   ~5.3M params
EfficientNet-B7 (phi=7)             →   ~66M params, substantially higher ImageNet accuracy,
                                          achieved by scaling depth, width, AND resolution together
```

### The efficiency half of "EfficientNet": depthwise separable convolutions

Scaling up naively would still be expensive per-layer, since a standard convolution's cost grows with both spatial filter size and the *product* of input and output channels (Lesson 02: `f * f * C_in * C_out` parameters per layer). EfficientNet's building block (originally introduced in MobileNet, and reused here) replaces a standard convolution with two cheaper steps:

```
Standard convolution:              Depthwise separable convolution:

  one f x f x C_in filter            Step 1 - Depthwise: one f x f filter PER input channel
  per output channel,                (no mixing across channels yet)
  C_out of them total
                                     Step 2 - Pointwise: a 1x1 convolution across all
  Cost: f*f*C_in*C_out               channels (mixes channels, no spatial extent)

                                     Cost: f*f*C_in (depthwise) + C_in*C_out (pointwise, 1x1)
```

For a 3×3 filter with `C_in = C_out = 256`:

```
Standard:              3*3*256*256 = 589,824 params
Depthwise separable:   3*3*256 + 256*256 = 2,304 + 65,536 = 67,840 params

Roughly 8.7x fewer parameters for a similar receptive field and channel mixing.
```

The idea: a standard convolution does two things at once (finds spatial patterns *and* mixes information across channels); depthwise separable convolution does them as two cheaper sequential steps instead, at a small cost to representational flexibility that empirically matters little in practice — this is exactly the same "factorize an expensive operation into cheaper pieces" instinct as VGG's stacked-3×3 trick (Lesson 13) and GoogLeNet's 1×1 bottleneck (Lesson 14), applied differently.

### MBConv: EfficientNet's core block

EfficientNet's building block, MBConv (Mobile Inverted Bottleneck Convolution), combines a depthwise separable convolution with an inverted bottleneck (expand channels with a 1×1 conv, apply the depthwise conv, then project back down with another 1×1 conv) and a squeeze-and-excitation mechanism (a small learned per-channel reweighting, letting the network emphasize more informative channels). The precise mechanics are beyond what this lesson derives in full, but the underlying instinct — factorize expensive operations, reuse bottleneck ideas from GoogLeNet/ResNet, and scale everything in a principled, balanced way — is a direct continuation of the ideas built up across Lessons 13-15.

### Why this mattered

EfficientNet-B0 through B7 achieved state-of-the-art ImageNet accuracy with 5-10x fewer parameters than comparably accurate prior architectures, by replacing ad hoc scaling decisions with a systematic, compute-aware recipe. This made high-accuracy models practically deployable on more constrained hardware (mobile devices, edge devices) — a direct consequence of depthwise separable convolutions' efficiency — while the compound scaling idea itself influenced how later architectures approach the question of "how do I make this model bigger without wasting the extra compute."

See `code/efficientnet_demo.py` for a from-scratch depthwise separable convolution with a parameter-count comparison against a standard convolution, and a simplified MBConv-style block.

## Exercises

1. Implement a depthwise separable convolution using `nn.Conv2d(groups=C_in)` for the depthwise step and a 1×1 `nn.Conv2d` for the pointwise step. Verify the output shape matches a standard convolution with the same input/output channels.
2. Reproduce the parameter-count comparison above (standard vs depthwise separable, 256→256 channels, 3×3 filter) and confirm the ~8.7x reduction.
3. Compute the compound scaling exponents' effect for `phi = 1, 2, 3` given illustrative values `alpha=1.2, beta=1.1, gamma=1.15`, and note how each dimension grows.
4. Implement a simplified MBConv block (expand → depthwise conv → project, with a skip connection when input/output shapes match) and trace its shape for a sample input.

## Key Terms

| Term | What it actually means |
|---|---|
| Compound scaling | Scaling a CNN's depth, width, and input resolution together in a fixed, balanced ratio, rather than scaling any one dimension alone |
| Depth scaling | Increasing a network's number of layers |
| Width scaling | Increasing the number of channels (filters) per layer |
| Resolution scaling | Increasing the input image size fed to the network |
| Depthwise separable convolution | A convolution factored into a per-channel spatial (depthwise) step and a channel-mixing 1x1 (pointwise) step, dramatically cheaper than a standard convolution |
| MBConv (Mobile Inverted Bottleneck Convolution) | EfficientNet's core building block, combining an inverted bottleneck, depthwise separable convolution, and squeeze-and-excitation channel reweighting |
