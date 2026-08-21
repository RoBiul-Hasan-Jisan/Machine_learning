# 13. VGG

## Learning Objectives

- Explain VGG's central idea: replacing large filters with stacks of small 3x3 filters
- Compute the receptive field and parameter savings of stacked 3x3 filters vs a single larger filter
- Implement VGG-16 in PyTorch using its repeated block structure

## The Problem

AlexNet (Lesson 12) used a mix of filter sizes — 11×11, 5×5, 3×3 — chosen somewhat ad hoc. VGG (Simonyan & Zisserman, 2014) asked a simpler question: what if you standardize on the smallest reasonable filter size (3×3) everywhere, and get depth and receptive field purely by stacking more of them? The answer, at the time, was a substantial accuracy improvement and a much simpler, more uniform architecture — VGG placed second in the 2014 ImageNet competition (behind GoogLeNet, Lesson 14) and became, for years afterward, one of the most widely used backbones for transfer learning (Lesson 17) precisely because of its clean, regular structure.

## The Concept

### The core idea: stack small filters instead of using one large filter

Two stacked 3×3 convolutional layers (stride 1, no pooling between them) have the same effective receptive field as a single 5×5 layer — a value derived directly from the output-size formula in Lesson 04 — but with fewer parameters and an extra nonlinearity in between.

```
Effective receptive field:

One 5x5 conv layer:              Two stacked 3x3 conv layers:
   5x5 = 25 weights                 (3x3 + 3x3) = 18 weights (per input/output channel pair)
   1 nonlinearity                    2 nonlinearities
   sees a 5x5 region of input        ALSO sees a 5x5 region of input
                                      (first 3x3 covers 3x3; second 3x3 over that
                                       output covers an effective 5x5 in the original input)
```

For `C` input and output channels, a single 5×5 layer costs `5 * 5 * C * C = 25C²` parameters; two stacked 3×3 layers cost `2 * (3 * 3 * C * C) = 18C²` — roughly 28% fewer parameters for the same effective receptive field, plus an extra ReLU (Lesson 06) between them, giving the network more representational capacity along the way (more nonlinear decision boundaries) than one big linear filter followed by a single activation could provide.

Three stacked 3×3 layers similarly match a single 7×7 layer's receptive field (`3*3*3=27` vs `7*7=49` weights per channel pair — about 45% fewer), which is the specific comparison VGG's original paper uses to argue for small, stacked filters as a near-strict improvement over larger ones.

### VGG's block-based design

VGG standardizes on a repeating pattern: several 3×3 conv layers (all with "same" padding, stride 1) followed by one 2×2 max pool that halves spatial size, repeated with the channel count doubling at each stage.

```
VGG-16 architecture:

Input: 224x224x3
    ↓
[Conv(64, 3x3)] x 2  → MaxPool(2x2)     → 112x112x64
[Conv(128, 3x3)] x 2 → MaxPool(2x2)     → 56x56x128
[Conv(256, 3x3)] x 3 → MaxPool(2x2)     → 28x28x256
[Conv(512, 3x3)] x 3 → MaxPool(2x2)     → 14x14x512
[Conv(512, 3x3)] x 3 → MaxPool(2x2)     → 7x7x512
    ↓
Flatten                                  → 25088
    ↓
FC(4096) → ReLU → Dropout
FC(4096) → ReLU → Dropout
FC(1000)
```

16 weight layers total (13 conv + 3 FC), hence "VGG-16." VGG-19 extends the same pattern with more conv layers per block (up to 4 in the deeper stages) for 19 weight layers total. Every convolutional layer in the entire network uses the same 3×3 filter, same padding, same stride — a strikingly uniform design compared to AlexNet's varied filter sizes.

### Why this uniform design was valuable

- **Simplicity**: one filter size, one padding rule, one pooling rule, repeated. Easy to describe, easy to implement, easy to reason about, easy to extend (VGG-11, VGG-13, VGG-16, VGG-19 are all the same pattern at different depths).
- **Depth as the main lever**: VGG's central empirical finding was that increasing depth alone, with everything else held simple and uniform, reliably improved accuracy up to a point — a finding that helped motivate the push toward even deeper networks, which then ran into the vanishing gradient and degradation problems that ResNet (Lesson 15) was specifically designed to solve.
- **Transferability**: VGG's learned features, particularly from VGG-16, turned out to transfer unusually well to other vision tasks. For years it was a default backbone choice for transfer learning (Lesson 17) and for tasks like object detection and style transfer, valued for its reliable, well-understood feature representations even as more efficient architectures (Lessons 14-16) were introduced.

### The cost of this design

VGG-16 has about 138 million parameters — more than AlexNet — the overwhelming majority again concentrated in the FC layers (Lesson 07's lesson about FC layers dominating parameter count applies here even more starkly). This makes VGG large and slow to train and store compared to later, more parameter-efficient architectures (Lessons 14-16), which is the main reason it's less commonly chosen for training from scratch today, even though it remains a solid, simple option for transfer learning.

See `code/vgg_demo.py` for a PyTorch implementation of VGG-16 built from a configurable block structure, a receptive-field/parameter comparison between one 5×5 layer and two stacked 3×3 layers, and a full shape/parameter trace.

## Exercises

1. Verify by hand that two stacked 3×3 conv layers (stride 1, no padding) produce the same output spatial size reduction as one 5×5 layer, using Lesson 04's output-size formula.
2. Compute the parameter count for a single 7×7 conv layer with `C=256` input and output channels, versus three stacked 3×3 layers with the same channel counts. Confirm the ~45% parameter savings claimed above.
3. Implement VGG-16 using a configurable list-based block builder (e.g. `cfg = [64, 64, 'M', 128, 128, 'M', ...]`) so VGG-11/13/19 can be built by changing only the config list.
4. Load `torchvision.models.vgg16(weights="IMAGENET1K_V1")` and compare its total parameter count against your from-scratch implementation.

## Key Terms

| Term | What it actually means |
|---|---|
| VGG | A 2014 CNN architecture (Simonyan & Zisserman) built entirely from stacked 3x3 convolutional layers, valued for its simplicity and transferability |
| Receptive field | The region of the original input that a given unit's value depends on, which grows as more layers are stacked |
| Stacked small filters | Using multiple small (e.g. 3x3) convolutional layers in sequence to match the receptive field of one larger filter, with fewer parameters and more nonlinearity |
| VGG-16 / VGG-19 | Specific VGG depths, named for their total count of weight layers (13 conv + 3 FC = 16; 16 conv + 3 FC = 19) |
