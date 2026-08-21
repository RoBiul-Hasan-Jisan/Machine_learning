# 11. LeNet

## Learning Objectives

- Describe LeNet-5's architecture and why it was designed the way it was
- Implement LeNet-5 in PyTorch and verify its layer-by-layer output shapes
- Situate LeNet historically as the first practical demonstration of the CNN pattern from Lessons 01-10

## The Problem

By the early 1990s, the pieces from this module's first ten lessons — convolution, pooling, backpropagation — were understood in principle, but not yet proven to work end-to-end on a real, practical task. LeNet-5 (Yann LeCun et al., 1998, building on earlier versions from 1989) was the architecture that proved it: trained to recognize handwritten digits for automated check reading, deployed commercially, and directly establishing the Conv → Pool → ... → FC pattern this whole module builds on.

## The Concept

### LeNet-5 architecture

```
Input: 32x32x1 (grayscale)
    ↓
Conv(6 filters, 5x5, stride 1)      → 28x28x6
    ↓
Avg Pool(2x2, stride 2)             → 14x14x6
    ↓
Conv(16 filters, 5x5, stride 1)     → 10x10x16
    ↓
Avg Pool(2x2, stride 2)             → 5x5x16
    ↓
Flatten                             → 400
    ↓
FC(120)
    ↓
FC(84)
    ↓
FC(10)                              → 10 class scores (digits 0-9)
```

Two convolutional blocks, each halving spatial size and increasing channel count — exactly the canonical pattern from Lesson 07, described here for the first time in its original, historical form. Note LeNet uses **average pooling**, not max pooling (Lesson 05) — max pooling wasn't yet the established default it later became. The original also used `tanh` activations rather than ReLU (Lesson 06), since ReLU wasn't popularized for deep networks until AlexNet, over a decade later (Lesson 12).

### Why this design

- **32×32 input, not 28×28** (MNIST's native size): LeNet's authors deliberately used a slightly larger input so a digit could be centered with margin on all sides, letting the first 5×5 filters fully "see" strokes near the image border without needing explicit padding (Lesson 04's padding concept existed, but LeNet's original design instead sidestepped the issue by construction).
- **5×5 filters**: larger than the 3×3 filters that later became standard (Lesson 13's VGG), reflecting an earlier design instinct to capture more context per filter rather than stacking many small filters.
- **Only 2 convolutional layers**: computationally, this was close to the practical limit for the hardware of the era. Depth (Lessons 13-15) became viable only as compute grew.
- **FC(120) → FC(84) → FC(10)**: a shrinking sequence of fully connected layers doing the final classification, following the same "flatten then FC" pattern from Lesson 07 — a pattern LeNet effectively originated for this task.

### Why LeNet still matters

LeNet-5 is small enough to fully understand and train in minutes on a laptop, which makes it the standard first architecture to implement when learning CNNs — every piece traces directly back to Lessons 02-10 with none of the additional complexity (skip connections, inception modules, batch norm) that later architectures introduce. It's also a genuine historical artifact: it was used commercially by banks to read handwritten check amounts throughout the 1990s, one of the earliest deployed applications of a trained neural network.

### LeNet in modern form

A "modern LeNet" swaps average pooling for max pooling and `tanh` for ReLU, incorporating the improvements from Lessons 05-06 while keeping the same overall shape — this is the version most tutorials (and this lesson's code) actually implement, since it trains faster and slightly more accurately than the 1998 original while remaining architecturally identical in spirit.

```python
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
```

See `code/lenet_demo.py` for the complete implementation, a layer-by-layer shape trace on a 32×32×1 input, a parameter count, and a short training run on a synthetic digit-like dataset.

## Exercises

1. Trace LeNet-5's output shape at every layer by hand for a 32×32×1 input, then verify with the PyTorch implementation.
2. Compute LeNet-5's total parameter count using the formulas from Lesson 07, and identify which layer contributes the most parameters.
3. Swap `AvgPool2d` for `MaxPool2d` (or vice versa) and compare training accuracy over a few epochs on a small dataset.
4. Modify LeNet to accept 28×28 input directly (MNIST's native size) by adjusting padding, and confirm the final flattened size and FC layers still work out correctly.

## Key Terms

| Term | What it actually means |
|---|---|
| LeNet-5 | The 1998 CNN architecture (LeCun et al.) that first demonstrated convolutional networks working practically on a real task (handwritten digit recognition) |
| tanh activation | A saturating S-shaped activation function used in the original LeNet, later largely superseded by ReLU (Lesson 06) |
| Modern LeNet | A common variant substituting max pooling and ReLU for the original's average pooling and tanh, otherwise architecturally identical |
