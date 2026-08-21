# 02. Convolution Operation

## Learning Objectives

- Compute a 2D convolution by hand for a small input and filter
- Implement 2D convolution (technically cross-correlation) from scratch with NumPy
- Explain the difference between mathematical convolution and the operation deep learning frameworks call "convolution"

## The Problem

Lesson 01 said a small filter "slides across the image" and detects patterns. This lesson makes that mechanical: exactly what arithmetic happens at each position, and why the result is a smaller grid of numbers called a feature map.

## The Concept

### The sliding-window dot product

At each position, take the filter, overlay it on a same-sized patch of the input, multiply element-wise, and sum. That single number becomes one output pixel. Slide the filter by one step (Lesson 04 covers stride) and repeat.

```
Input (5x5):                Filter (3x3):
1  2  3  0  1                1  0  1
0  1  2  3  0                0  1  0
1  0  1  2  0                1  0  1
2  3  0  1  2
0  1  2  0  1

Output position (0,0): overlay filter on the top-left 3x3 patch of the input

Patch:            Filter:            Element-wise product:      Sum:
1 2 3             1 0 1              1*1  2*0  3*1              1+0+3
0 1 2      x      0 1 0       =      0*0  1*1  2*0        =     0+1+0    =  7
1 0 1             1 0 1              1*1  0*0  1*1              1+0+1

Output[0,0] = 7
```

Slide the filter one position to the right and repeat for `Output[0,1]`, and so on across every valid position. The result is the **feature map** (also called the activation map).

### Output size

For an `n × n` input and an `f × f` filter with stride 1 and no padding:

```
output_size = n - f + 1
```

A 5×5 input with a 3×3 filter gives a 3×3 output. The output shrinks because the filter can't be centered on border pixels without running off the edge — this is exactly what padding (Lesson 04) exists to control.

### Convolution vs cross-correlation

Mathematically, "convolution" flips the filter (180 degrees) before sliding it; the operation just shown, without flipping, is technically called **cross-correlation**. Every deep learning framework (PyTorch, TensorFlow) implements cross-correlation and calls it "convolution" anyway. This doesn't matter for learning: since the filter's weights are learned from data, a network trained with cross-correlation just learns the (already-flipped) equivalent of a true convolutional filter. This lesson — and the rest of this module — uses "convolution" in the deep-learning sense (no flipping), matching every framework you'll actually use.

### Multi-channel input

A color image has 3 channels (R, G, B). A filter for it is not `f × f`, but `f × f × 3` — one `f × f` slice per input channel. The dot product sums over the filter's height, width, *and* channel dimensions, producing a single scalar per spatial position, same as before:

```
Input:  H x W x 3
Filter: f x f x 3
Output at each position: sum over all f*f*3 products  -> ONE number

So one filter, regardless of how many input channels it covers,
always produces one 2D output feature map.
```

### Multiple filters

A convolutional layer uses many filters, not one. Each filter produces its own 2D feature map, and the layer's output stacks them along a new channel dimension:

```
Layer with K filters, each f x f x C_in:
  Input:  H x W x C_in
  Output: H_out x W_out x K       (K feature maps, one per filter)
```

This is why layers deeper in a CNN have more channels but smaller spatial dimensions — each layer trades spatial resolution (via pooling/stride) for a richer set of learned feature maps.

### Implementing it from scratch

```python
import numpy as np

def convolve2d(input_matrix, kernel):
    """Single-channel 2D convolution (cross-correlation), stride 1, no padding."""
    ih, iw = input_matrix.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1

    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = input_matrix[i:i + kh, j:j + kw]
            output[i, j] = np.sum(patch * kernel)
    return output
```

This nested-loop version is intentionally simple and readable; it is far slower than a real framework's convolution, which uses vectorized matrix multiplication (the `im2col` trick) or FFT-based methods. See `code/convolution_from_scratch.py` for the naive version, a vectorized `im2col`-based version, and a comparison against `torch.nn.functional.conv2d` to confirm they agree numerically.

## Exercises

1. Compute the full 3×3 output feature map for the 5×5 input and 3×3 filter shown above, by hand or with `convolve2d`.
2. Extend `convolve2d` to handle multi-channel input (shape `H x W x C`) and a matching multi-channel filter (`f x f x C`), summing over channels.
3. Implement a layer that applies K filters to a multi-channel input and stacks the results, then confirm the output shape matches `(H_out, W_out, K)`.
4. Benchmark the naive triple-loop `convolve2d` against `torch.nn.functional.conv2d` on a 224×224 input with a 3×3 filter. Report the speed difference and explain why frameworks avoid the naive approach.

## Key Terms

| Term | What it actually means |
|---|---|
| Convolution (deep learning sense) | Sliding a filter across an input and computing an element-wise product and sum at each position, without flipping the filter (technically cross-correlation) |
| Filter / kernel | The small grid of learnable weights that slides across the input to produce a feature map |
| Feature map / activation map | The output of applying one filter across the input |
| im2col | A technique that reshapes convolution into a single large matrix multiplication for speed, used internally by deep learning frameworks |
