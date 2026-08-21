# 15. ResNet

## Learning Objectives

- Explain the degradation problem that motivated ResNet and why it isn't the same as overfitting
- Implement a residual block with a skip connection and explain why it makes very deep networks trainable
- Build ResNet-18/34-style networks from stacked residual blocks in PyTorch

## The Problem

VGG (Lesson 13) showed depth helps, up to a point. But researchers found that simply stacking more layers eventually made networks perform *worse* — and not because of overfitting (which would show as low training error but high test error). Very deep plain networks had **higher training error** than shallower ones. Something about training itself was breaking down as networks got deeper, a phenomenon called the **degradation problem**. ResNet (He et al., 2015) diagnosed and fixed this with one of the most influential ideas in deep learning: the skip connection.

## The Concept

### The degradation problem

If you take a well-trained shallow network and add more layers on top, in principle those extra layers could just learn the identity function (pass their input through unchanged), and the deeper network should never perform *worse* than the shallow one — at worst, equally well. In practice, plain (non-residual) deep networks struggled to learn even this trivial identity mapping through many stacked nonlinear layers, and training error actually rose with depth beyond a certain point. This wasn't a capacity problem (deeper networks have strictly more representational power) — it was an **optimization** problem: gradient descent, through many stacked nonlinear layers, was failing to find even the "do nothing extra" solution that was clearly achievable.

### The fix: skip connections (residual connections)

Instead of asking a block of layers to learn the desired output `H(x)` directly, ResNet reframes the target: let the block learn the *residual* `F(x) = H(x) - x`, and add the original input back in:

```
Plain block:                          Residual block:

    x                                      x
    |                                      |----------------\
  [layers]                              [layers]              |  (identity skip
    |                                      |                   |   connection)
  H(x)                                   F(x)                  |
                                           |                    |
                                           +--------------------/
                                           |
                                        F(x) + x  =  H(x)
```

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x              # the skip connection
        return self.relu(out)
```

If the identity mapping really is the best thing for this block to do, the block only needs to learn `F(x) = 0` — pushing all the block's weights toward zero — which is a far easier optimization target for gradient descent to find than learning an exact identity mapping through a stack of nonlinear transformations. The skip connection doesn't add any new representational power in principle (a plain network could theoretically express the same functions), but it makes the *easy* solutions easy to reach, which turns out to matter enormously in practice.

### Why this also helps gradients flow

During backpropagation (Lesson 09), the gradient with respect to `x` gets a direct path through the `+x` addition — the derivative of `F(x) + x` with respect to `x` includes a `+1` term from the identity, in addition to whatever gradient flows back through `F`. In a deep plain network, the gradient reaching an early layer is a product of many layers' Jacobians, and that product can shrink toward zero or blow up, depending on initialization — a compounding instability that gets worse with depth. The skip connection's guaranteed `+1` term means every residual block contributes a stable, well-scaled path for gradient, on top of whatever the block's own weights contribute — directly counteracting the vanishing/exploding gradient problem (Lesson 06) that gets worse as plain networks get deeper.

### Handling shape mismatches

The skip connection `out = F(x) + x` requires `F(x)` and `x` to have the same shape. When a block changes the number of channels or downsamples spatially (via a stride-2 conv), the skip path needs a matching transformation — typically a 1×1 convolution (Lesson 14) with the same stride, used purely to reshape the identity path, not to extract features:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)
```

### ResNet's overall structure

ResNet-18/34 stack residual blocks in stages, doubling channels and halving spatial size at the start of each stage (using a stride-2 block, with its matching 1×1 shortcut), similar in spirit to VGG's stage structure but with every block wrapped in a skip connection:

```
Input → Conv/Pool stem
    → Stage 1: residual blocks, 64 channels
    → Stage 2: residual blocks, 128 channels (first block strides down)
    → Stage 3: residual blocks, 256 channels (first block strides down)
    → Stage 4: residual blocks, 512 channels (first block strides down)
    → Global Average Pool → FC(num_classes)
```

Deeper variants (ResNet-50/101/152) use a **bottleneck block** (1×1 → 3×3 → 1×1, echoing GoogLeNet's Inception bottleneck idea from Lesson 14) instead of the plain two-3×3-conv block, reducing computational cost enough to make hundreds of layers practical. ResNet-152, remarkably, has *fewer* parameters than VGG-16 despite being roughly 9x deeper — depth without a linear parameter explosion, made possible by aggressive channel bottlenecking plus global average pooling.

### Why this mattered

ResNet made networks with over 100 layers trainable and reliably better than shallower ones — before ResNet, a 100+ layer plain network would have been essentially untrainable due to the degradation problem. Skip connections subsequently became a near-universal building block, adopted far beyond image classification — in object detection, segmentation, and eventually as a core component of the Transformer architecture that underlies modern language models, which uses the identical residual-connection idea around each of its attention and feed-forward sublayers.

See `code/resnet_demo.py` for a complete residual block implementation with shape-mismatch handling, a demonstration that gradients flow through many stacked residual blocks without vanishing, and a small ResNet-style network built by stacking blocks into stages.

## Exercises

1. Implement the basic residual block above and confirm `F(x) + x` requires the shortcut path only when `stride != 1` or channels change; verify both cases produce correctly shaped output.
2. Train a very deep (20+ layer) plain (non-residual) network and an equally deep residual network on the same small dataset for the same number of epochs. Compare final training loss to illustrate the degradation problem and its fix.
3. Implement a bottleneck residual block (1×1 → 3×3 → 1×1) and compare its parameter count to a plain two-3×3-conv block for the same input/output channel counts.
4. Load `torchvision.models.resnet18(weights="IMAGENET1K_V1")` and inspect its stage structure with `print(model)`, matching what you see to the stage diagram above.

## Key Terms

| Term | What it actually means |
|---|---|
| Degradation problem | The empirical finding that very deep plain networks have higher *training* error than shallower ones, indicating an optimization failure rather than overfitting |
| Skip connection (residual connection) | A connection that adds a block's input directly to its output, letting the block learn a residual correction rather than a full transformation |
| Residual block | A group of layers wrapped in a skip connection, the basic building unit of ResNet |
| Bottleneck block | A residual block variant using 1x1 convolutions to reduce then restore channel count around a central 3x3 convolution, reducing computational cost in very deep ResNets |
| ResNet | A CNN architecture (He et al., 2015) built from stacked residual blocks, which made networks with 100+ layers reliably trainable |
