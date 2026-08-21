# 04. Stride and Padding

## Learning Objectives

- Compute output dimensions for a convolution given input size, filter size, stride, and padding
- Explain what stride and padding each control, and why both are needed in practice
- Implement strided and padded convolution from scratch

## The Problem

Lesson 02's convolution always moved the filter one step at a time and always shrank the output (an `n × n` input with an `f × f` filter gives `(n-f+1) × (n-f+1)` output). Two practical problems follow: shrinking output size limits how many layers you can stack before running out of spatial dimensions, and border pixels get used in far fewer output computations than center pixels, effectively under-weighting the edges of the image. Stride and padding are the two knobs that address these.

## The Concept

### Stride: how far the filter moves each step

Stride controls the step size as the filter slides. Stride 1 (Lesson 02's default) moves one pixel at a time. Stride 2 skips every other position, producing a roughly half-sized output.

```
Stride 1: filter visits every position    → output ≈ input size
Stride 2: filter visits every other position → output ≈ half the input size
```

The general output size formula, for an `n × n` input, `f × f` filter, and stride `s` (no padding):

```
output_size = floor((n - f) / s) + 1
```

Example: `n=7, f=3, s=2` → `floor((7-3)/2) + 1 = floor(2) + 1 = 3`. A 7×7 input becomes a 3×3 output.

Stride is a cheap way to downsample: a stride-2 convolution does the work of both feature extraction and spatial reduction in one operation, which is why several modern architectures (Lesson 15's ResNet, for instance) use strided convolutions instead of separate pooling layers to reduce spatial size.

### Padding: adding a border before convolving

Padding adds extra pixels (usually zeros — "zero padding") around the input border before convolving, so border pixels get used in as many output computations as center pixels, and so the output size can be controlled independently of stride.

```
Original 5x5 input:              Padded to 7x7 (1-pixel zero border):

1 2 3 0 1                        0 0 0 0 0 0 0
0 1 2 3 0                        0 1 2 3 0 1 0
1 0 1 2 0        pad=1    →      0 0 1 2 3 0 0
2 3 0 1 2                        0 1 0 1 2 0 0
0 1 2 0 1                        0 2 3 0 1 2 0
                                  0 0 1 2 0 1 0
                                  0 0 0 0 0 0 0
```

With padding `p` added on all sides, the output size formula becomes:

```
output_size = floor((n + 2p - f) / s) + 1
```

### "Valid" vs "same" padding

Deep learning frameworks expose two common padding presets:

| Mode | Padding amount | Effect |
|---|---|---|
| **Valid** | `p = 0` | No padding. Output shrinks with every layer, as in Lesson 02 |
| **Same** | `p` chosen so output size equals input size (for stride 1) | Output spatial size stays the same as the input |

For stride 1, "same" padding requires `p = (f - 1) / 2`, which is a whole number whenever `f` is odd — one reason 3×3 and 5×5 filters (odd sizes) are far more common than 2×2 or 4×4 in practice: they allow a clean, symmetric "same" padding.

```python
# f=3 filter, stride 1, "same" padding:
p = (3 - 1) // 2   # = 1  (one pixel of padding on each side)
```

### Why padding matters beyond keeping sizes the same

Without padding, a pixel at the very corner of the image participates in only one output computation across the whole feature map, while a center pixel participates in `f × f` of them (assuming stride 1). Repeated across many layers, this means information near the border gets used far less than information in the center, and the network implicitly treats edge content as less important — usually not what you want. Padding fixes this by ensuring the filter can be centered on every original pixel, including those at the border.

### Putting stride and padding together

Most real architectures use "same" padding within a block (to stack many convolutional layers without shrinking) and use stride 2 (with or without extra padding) only at specific points where deliberate downsampling is wanted — echoing the pattern from Lesson 07's typical CNN architecture, where pooling or strided convolutions periodically halve spatial dimensions while the number of channels grows.

See `code/stride_padding_demo.py` for a runnable implementation of strided, padded convolution from scratch, with output-size verification against the formula and against `torch.nn.Conv2d`.

## Exercises

1. Compute the output size for `n=32, f=5, s=1, p=0` (valid) and `n=32, f=5, s=1, p=2` (same). Verify both against the formula.
2. Compute the "same" padding amount for filter sizes 3, 5, and 7 with stride 1. Explain why an even filter size (e.g. 4) does not have a clean symmetric "same" padding.
3. Implement padded, strided convolution from scratch by first constructing the zero-padded input array, then running strided convolution over it. Confirm the output size matches the formula for several `(n, f, s, p)` combinations.
4. Using `torch.nn.Conv2d`, build a stack of 5 convolutional layers with "same" padding and stride 1, and confirm the spatial size is unchanged after all 5 layers. Then replace one layer with stride 2 and confirm the size roughly halves at that point.

## Key Terms

| Term | What it actually means |
|---|---|
| Stride | The number of pixels the filter moves between each convolution step |
| Padding | Extra border pixels (usually zeros) added around the input before convolving |
| Valid padding | No padding (`p=0`); output shrinks with filter size |
| Same padding | Padding chosen so the output spatial size matches the input (for stride 1) |
| Zero padding | The common convention of filling padding pixels with zeros |
