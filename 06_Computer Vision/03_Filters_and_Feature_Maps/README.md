# 03. Filters and Feature Maps

## Learning Objectives

- Explain what a filter "detects" and how that connects to its weight pattern
- Apply classic hand-designed filters (edge detectors, blur, sharpen) and interpret the resulting feature maps
- Understand what it means for a CNN to "learn" filters, as opposed to hand-designing them

## The Problem

Lesson 02 established the mechanics: a filter slides across an input and produces a feature map. This lesson asks the more interesting question: what makes a filter's output actually mean something? Why does a particular 3×3 grid of numbers detect vertical edges, and how does a CNN end up with filters like that without anyone specifying them by hand?

## The Concept

### A filter's weights encode what it looks for

A filter produces a large output wherever the input locally matches its weight pattern, because convolution is a dot product — and a dot product between two vectors is largest when they point in a similar direction.

**Vertical edge detector (Sobel-style):**

```
-1  0  1
-2  0  2
-1  0  1
```

This filter is negative on the left, positive on the right. Its dot product with a patch is large (positive) when the patch is dark on the left and bright on the right — a vertical edge going from dark to light. It's near zero on a flat, uniform patch, since the negative and positive weights cancel out against similar pixel values.

**Horizontal edge detector:** the same idea, rotated 90 degrees:

```
-1 -2 -1
 0  0  0
 1  2  1
```

**Blur (box filter):** every weight equal and positive, normalized to sum to 1. This averages a patch, smoothing out sharp changes:

```
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

**Sharpen:** a large positive center weight surrounded by negative weights, which amplifies the difference between a pixel and its neighbors:

```
 0  -1   0
-1   5  -1
 0  -1   0
```

These are all hand-designed — decades of classical computer vision (before deep learning) consisted largely of designing filters like these by hand for specific tasks.

### Reading a feature map

The output of applying a filter is a 2D grid where each value says "how strongly does this filter's pattern match here." High values (in magnitude) mark locations where the input locally resembles the filter's pattern; values near zero mark locations where it doesn't. Visualizing a feature map as a grayscale image after applying a vertical edge filter to a photo will show bright lines exactly where vertical edges exist in the original — the filter has turned "where are the vertical edges" into an explicit, visual answer.

### Learned filters vs hand-designed filters

Classical computer vision hand-designs filters like the ones above for a specific task. CNNs instead **learn** the filter weights via gradient descent (Lesson 09), starting from small random values and updating them to reduce the loss on a training set.

The remarkable empirical finding, first clearly visualized in AlexNet (Lesson 12) and studied extensively since: filters learned by the first convolutional layer of a CNN trained on natural images consistently converge to something close to edge detectors and color-opponent blobs — very similar to the hand-designed filters above — *without ever being told to*. The network discovers, from gradient descent alone, that edges and color gradients are useful building blocks for recognizing objects.

Deeper layers' filters are harder to interpret directly (they operate on feature maps, not raw pixels), but techniques like activation maximization show they respond to increasingly complex patterns: textures in the middle layers, object parts (wheels, eyes, windows) in later layers.

### One filter, one feature map; many filters, many feature maps

A convolutional layer with 32 filters produces 32 feature maps, stacked into a `H × W × 32` output (Lesson 02). Each of the 32 filters can specialize in detecting a different pattern — one for vertical edges, one for a particular color transition, one for a diagonal texture — and the layer's full output is the combined "read-out" of all 32 detectors at every spatial position.

See `code/filters_demo.py` for a runnable example applying classic hand-designed filters to a synthetic image and visualizing the resulting feature maps, plus a small trained CNN whose learned first-layer filters can be inspected directly.

## Exercises

1. Apply the vertical and horizontal edge filters above to a synthetic image containing a vertical bar and a horizontal bar. Confirm each filter produces a strong response only on its corresponding edge.
2. Design your own 3×3 filter meant to detect a diagonal edge (top-left to bottom-right). Test it on a synthetic diagonal-edge image and check whether it behaves as intended.
3. Train a tiny CNN (2-3 conv layers) on a small image dataset (e.g. a subset of MNIST) for a few epochs, then visualize the first layer's learned filters as grayscale grids. Compare them qualitatively to the hand-designed edge/blur filters above.
4. Apply the same filter to a flat, uniform-color region of an image and to a region with a sharp edge. Explain numerically why the flat region produces a near-zero response.

## Key Terms

| Term | What it actually means |
|---|---|
| Edge detector | A filter whose weight pattern produces a large response at sharp intensity transitions in the input |
| Feature map | The 2D grid of responses produced by sliding one filter across the input; high values mark locations that match the filter's pattern |
| Hand-designed filter | A filter whose weights are set manually based on domain knowledge (classical computer vision), rather than learned from data |
| Learned filter | A filter whose weights are initialized randomly and updated via gradient descent to minimize a training loss |
| Activation maximization | A visualization technique that finds (or synthesizes) an input maximizing a given filter's response, used to interpret what deep filters detect |
