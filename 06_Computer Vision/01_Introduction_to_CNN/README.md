# 01. Introduction to CNN

## Learning Objectives

- Explain why fully connected networks scale poorly to images
- Identify the three core ideas CNNs add: local receptive fields, weight sharing, and hierarchical feature learning
- Recognize the standard building blocks of a CNN and where this module covers each one

## The Problem

A 224×224 RGB image has 224 × 224 × 3 = 150,528 input values. A fully connected layer mapping that to just 1,000 hidden units needs over 150 million weights — in one layer, before the network has learned anything about images specifically. That's expensive to train, easy to overfit, and throws away something important: a fully connected layer treats pixel (0,0) and pixel (200,200) as unrelated inputs, learning a separate weight for every pixel-position combination, even though the same edge or texture can appear anywhere in the image.

Convolutional Neural Networks (CNNs) fix both problems with the same idea.

## The Concept

### The core idea: local, shared, hierarchical

A CNN replaces "one weight per input pixel" with a small filter (e.g. 3×3 or 5×5 weights) that slides across the entire image, reusing the same weights at every position.

```
Fully connected:  every output unit has its own weight for every input pixel
                   → huge parameter count, no notion of "nearby pixels matter more"

Convolutional:     a small filter (say 3x3 = 9 weights) slides across the image,
                   the SAME 9 weights are reused at every position
                   → far fewer parameters, and the filter becomes a detector
                     for one local pattern (an edge, a corner, a texture)
                     wherever it appears in the image
```

This rests on three ideas that this module builds up one at a time:

1. **Local receptive fields** — each output value depends only on a small neighborhood of the input, not the entire image. This matches how visual patterns actually work: an edge is a local phenomenon, not a global one. (Lesson 02)
2. **Weight sharing (parameter sharing)** — the same filter is applied at every spatial position, so a pattern learned in one part of the image is automatically detected anywhere else it appears. This is what makes CNNs translation-equivariant: shift the input, and the feature map shifts by the same amount. (Lessons 02-03)
3. **Hierarchical feature learning** — stacking convolutional layers lets early layers detect simple patterns (edges, colors, gradients) and later layers combine those into increasingly complex, larger-scale patterns (textures, parts, objects). (Lesson 07)

### What a CNN looks like end to end

```
Input Image
    ↓
[Conv → Activation → Pool]  ×  N     (feature extraction: Lessons 02-07)
    ↓
Flatten
    ↓
Fully Connected Layer(s)              (classification head)
    ↓
Output (class scores)
```

Early layers see raw pixels and learn simple, generic filters — edge detectors, color blobs. Deeper layers combine those into shapes, textures, and eventually object parts. By the time you reach the fully connected head, the network is working with a compact, highly informative representation of "what's in this image," not raw pixels.

### Where this module goes from here

| Lessons | What they cover |
|---|---|
| 02-06 | The mechanics of a single convolutional layer: the convolution operation, filters/feature maps, stride/padding, pooling, activation functions |
| 07-10 | Assembling layers into a full architecture, and how forward/backward propagation and training actually work |
| 11-16 | The landmark architectures that defined the field, in chronological order: LeNet → AlexNet → VGG → GoogLeNet → ResNet → EfficientNet |
| 17-18 | Practical techniques for using CNNs effectively with limited data: transfer learning and data augmentation |
| 19 | End-to-end projects that combine everything |

### Why CNNs, still, in the era of Vision Transformers

Vision Transformers (ViTs) have matched or exceeded CNNs on large-scale image benchmarks, but CNNs remain the practical default in many settings:
- They need less data to train from scratch — the built-in local/shared-weight assumptions act as a strong, useful bias for images.
- They're cheaper to run at the resolutions and hardware most production systems actually use.
- The landmark architectures (Lessons 11-16) are still the backbone of countless deployed vision systems, and transfer learning from a pretrained CNN (Lesson 17) remains one of the fastest paths to a working model on a new dataset.

## Exercises

1. Compute the number of parameters in a fully connected layer mapping a 32×32×3 image to a 500-unit hidden layer. Compare that to a convolutional layer using 16 filters of size 3×3×3. Explain the size difference in terms of local receptive fields and weight sharing.
2. Find three examples of visual patterns (edges, corners, textures) in a photo of your choice, and describe informally why detecting them doesn't require looking at the whole image at once.
3. Sketch (on paper or in a diagram) what you'd expect an early-layer filter to detect versus a late-layer filter, in a network trained to classify dogs vs cats.

## Key Terms

| Term | What it actually means |
|---|---|
| Convolutional Neural Network (CNN) | A neural network architecture using convolutional layers with local, shared-weight filters, suited to grid-structured data like images |
| Local receptive field | The small region of the input that a single output unit depends on |
| Weight sharing (parameter sharing) | Using the same filter weights at every spatial position, rather than a separate weight per position |
| Translation equivariance | The property that shifting the input shifts the output feature map by the same amount, a direct consequence of weight sharing |
| Feature hierarchy | The pattern where early CNN layers learn simple, generic features and deeper layers combine them into complex, task-specific features |
