# 07. CNN Architecture

## Learning Objectives

- Assemble convolution, activation, pooling, and fully connected layers into a complete CNN
- Explain the typical pattern of shrinking spatial size and growing channel depth through a network
- Compute the parameter count and output shape of a full CNN, layer by layer

## The Problem

Lessons 02-06 covered each building block in isolation: convolution, filters, stride/padding, pooling, activation functions. This lesson assembles them into the shape every classic CNN actually takes, and shows how to reason about a full network's shapes and parameter counts before ever running it.

## The Concept

### The canonical pattern

```
Input Image (H x W x 3)
        ↓
[Conv → Activation] × 1-2  →  Pool     ]
[Conv → Activation] × 1-2  →  Pool     ]  repeated blocks:
[Conv → Activation] × 1-2  →  Pool     ]  spatial size shrinks, channels grow
        ↓
Flatten
        ↓
Fully Connected → Activation
        ↓
Fully Connected (output layer)
        ↓
Softmax (classification) or raw scores
```

Each block typically **halves spatial dimensions** (via pooling or a stride-2 conv) while **doubling the number of channels** (via more filters in the next conv layer). The intuition: as spatial detail becomes less necessary (you've already extracted the local patterns worth keeping), the network trades that resolution for a richer set of feature channels describing what's present at each remaining location.

```
Example channel/size progression (like a simplified VGG-style network):

224x224x3   (input)
  ↓ conv block (64 filters)
224x224x64
  ↓ pool
112x112x64
  ↓ conv block (128 filters)
112x112x128
  ↓ pool
56x56x128
  ↓ conv block (256 filters)
56x56x256
  ↓ pool
28x28x256
  ↓ flatten + FC layers
  ↓
output (e.g. 1000 classes)
```

### Why flatten, and why FC layers at the end

After the convolutional blocks, the feature maps hold spatially-organized information ("there's a curved edge around here, a texture over there"). The final classification decision, though, needs to combine information from across the *entire* image — "wheel here + windshield there + headlights there = car" — which is exactly what a fully connected layer does: every output unit sees every input value, with no locality restriction. `Flatten` simply reshapes the final `H × W × C` feature map into a single long vector so it can feed into a standard fully connected layer.

Some modern architectures (Lesson 14's GoogLeNet, Lesson 15's ResNet) replace large FC layers with **global average pooling** (Lesson 05) before a much smaller final FC layer — averaging each channel's feature map down to one number produces a compact summary with far fewer parameters than flattening a large spatial grid, while still combining information globally.

### Computing output shapes layer by layer

Given an architecture spec, you can compute every intermediate shape using the formulas from Lessons 02-05, without running any code:

```
Layer                          Output shape           Formula used
Input                          224 x 224 x 3           -
Conv(64 filters, 3x3, same)    224 x 224 x 64          same padding -> size unchanged, channels = filters
MaxPool(2x2, stride 2)         112 x 112 x 64           (224-2)/2 + 1 = 112
Conv(128 filters, 3x3, same)   112 x 112 x 128          same padding -> size unchanged
MaxPool(2x2, stride 2)         56 x 56 x 128            (112-2)/2 + 1 = 56
Flatten                        401408                   56 * 56 * 128
FC(256 units)                  256                      -
FC(10 units, output)           10                       -
```

### Computing parameter counts

A convolutional layer's parameter count depends only on the filter size and channel counts — **not** on the spatial size of the input, which is exactly the weight-sharing payoff from Lesson 01:

```
Conv layer params = (f * f * C_in + 1) * C_out
                                  ^
                            +1 for the bias term per filter

Example: Conv(64 filters, 3x3, input has 3 channels)
  params = (3 * 3 * 3 + 1) * 64 = (27 + 1) * 64 = 1,792
```

A fully connected layer's parameter count depends on both its input and output size, and grows fast:

```
FC layer params = (input_size + 1) * output_size

Example: FC layer from a 401,408-unit flattened input to 256 units
  params = (401,408 + 1) * 256 = 102,760,704   <- over 100 million parameters!
```

This single FC layer has vastly more parameters than every convolutional layer before it combined — a big part of why architectures increasingly favor global average pooling (small flattened size) over flattening a large spatial grid directly into a big FC layer.

### Design decisions this pattern leaves open

The lessons ahead (11-16) are essentially a history of different answers to: how many conv layers per block, how wide should the filters be, when to downsample, how deep can you go, and how do you keep training stable as depth increases. Each landmark architecture is a different, historically important answer to those questions.

See `code/architecture_demo.py` for a runnable CNN built with `torch.nn`, including automatic shape verification at every layer and a parameter count breakdown matching the formulas above.

## Exercises

1. Given an input of 32×32×3 and a network of `[Conv(32, 3x3, same) → Pool(2x2) → Conv(64, 3x3, same) → Pool(2x2) → Flatten → FC(128) → FC(10)]`, compute every intermediate shape by hand, then verify with `torch.nn`.
2. Compute the parameter count for each layer in the network above, and identify which single layer contributes the most parameters.
3. Replace the final Flatten + FC(128) with global average pooling before the last FC(10) layer. Recompute the parameter count and compare it to the flattened version.
4. Design a CNN for 64×64×3 input that ends with a spatial size of 4×4 before flattening, using only 2×2 max pooling layers. How many pooling layers are needed?

## Key Terms

| Term | What it actually means |
|---|---|
| Conv block | A group of one or more convolutional + activation layers, typically followed by a pooling or stride-2 layer |
| Flatten | Reshaping a multi-dimensional feature map into a single vector, to feed into a fully connected layer |
| Global average pooling | Averaging each channel's entire spatial feature map to a single number, used as a parameter-efficient alternative to flattening |
| Parameter count | The total number of learnable weights (and biases) in a layer or network, which for conv layers depends only on filter size and channel counts, not spatial size |
