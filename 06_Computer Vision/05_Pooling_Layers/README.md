# 05. Pooling Layers

## Learning Objectives

- Explain what pooling accomplishes that stride alone doesn't, and why it has no learnable parameters
- Implement max pooling and average pooling from scratch
- Compare pooling to strided convolution and know when architectures favor one over the other

## The Problem

After a few convolutional layers, feature maps still carry a lot of spatial detail — often more than the task actually needs. A cat's ear detector firing at pixel (40, 60) versus (41, 60) is essentially the same piece of information for "is there a cat in this image," but treating them as different signals wastes computation and makes the network needlessly sensitive to tiny shifts. Pooling summarizes a local neighborhood into one value, reducing spatial size while keeping the most relevant information.

## The Concept

### Max pooling: keep the strongest signal

Max pooling slides a window (typically 2×2) across each feature map and keeps only the maximum value in each window.

```
Input feature map (4x4):        Max pool, 2x2 window, stride 2:

1  3  2  4                      3  4
5  6  1  2          →           6  4
0  1  8  3
2  4  1  1                      4  8
```

Top-left 2×2 window `[1,3,5,6]` → max is 6. Top-right window `[2,4,1,2]` → max is 4. And so on. The intuition: if a filter is detecting "is there an edge here," the exact pixel position within a small neighborhood usually matters less than whether the edge is present *somewhere* in that neighborhood — max pooling keeps that "somewhere" signal and discards the precise position.

### Average pooling: smooth summary

Same sliding-window idea, but takes the mean instead of the max:

```
Same input, average pool, 2x2 window, stride 2:

(1+3+5+6)/4 = 3.75      (2+4+1+2)/4 = 2.25
(0+1+2+4)/4 = 1.75      (8+3+1+1)/4 = 3.25
```

Average pooling produces a smoother summary and is less prone to being dominated by a single outlier activation, but tends to dilute sparse strong signals (a single very active pixel gets averaged down with its quieter neighbors). Max pooling is the more common default in classic CNN architectures for this reason: it preserves "was this feature present" more sharply. Average pooling reappears at the very end of many modern architectures as **global average pooling** — averaging an entire feature map down to a single number per channel, replacing large fully connected layers before the output (used in GoogLeNet, Lesson 14, and ResNet, Lesson 15).

### Output size

Pooling uses the exact same size formula as convolution (Lesson 04), just with a "filter" that has no learnable weights:

```
output_size = floor((n - f) / s) + 1
```

The most common configuration is `f=2, s=2` (non-overlapping 2×2 windows), which exactly halves both spatial dimensions.

### Pooling has no learnable parameters

This is the detail that distinguishes it from convolution: a pooling layer computes a fixed function (max or mean) of its input window — there's nothing to learn, no weights, no bias. This makes pooling cheap (no extra parameters to train, negligible compute) and gives CNNs a small amount of built-in translation invariance: a feature shifted by a pixel or two within a pooling window still produces the same max (or nearly the same average) after pooling.

### Pooling vs strided convolution

Pooling and stride-2 convolution both reduce spatial size, but they aren't redundant:

| | Pooling | Strided convolution |
|---|---|---|
| Learnable? | No | Yes |
| What it does | Fixed summary (max/mean) of a window | Learned filter applied at intervals |
| Adds parameters? | No | Yes |
| Common use | Downsampling between conv blocks (classic CNNs) | Downsampling *and* feature extraction in one step (modern architectures, e.g. ResNet) |

Some architectures use both; some (like many ResNet variants) largely replace pooling with strided convolutions, on the reasoning that letting the network *learn* how to downsample can outperform a fixed max/average rule. Both remain in wide use — this module's architecture lessons (11-16) point out which each network uses.

See `code/pooling_demo.py` for a from-scratch max and average pooling implementation, verified against `torch.nn.MaxPool2d` / `torch.nn.AvgPool2d`, plus a small demo of pooling's translation-robustness.

## Exercises

1. Compute max pooling and average pooling by hand for the 4×4 example above using a 2×2 window and stride 1 instead of stride 2 (overlapping windows). Compare the output size to the stride-2 case.
2. Implement `max_pool2d` and `avg_pool2d` from scratch and confirm they match `torch.nn.MaxPool2d` / `torch.nn.AvgPool2d` numerically on random input.
3. Create a small feature map with one strong "active" pixel among mostly-zero neighbors. Apply max pooling and average pooling and compare how much of the strong signal survives in each case.
4. Shift a small test pattern by 1 pixel and compare the max-pooled output before and after the shift. Show that max pooling makes the output partially (not fully) robust to the shift.

## Key Terms

| Term | What it actually means |
|---|---|
| Max pooling | Downsampling by taking the maximum value in each local window of a feature map |
| Average pooling | Downsampling by taking the mean value in each local window of a feature map |
| Global average pooling | Averaging an entire feature map down to a single value per channel, often replacing large fully connected layers |
| Translation invariance (partial) | The property that small input shifts produce little or no change in pooled output, due to the pooling window absorbing minor position changes |
