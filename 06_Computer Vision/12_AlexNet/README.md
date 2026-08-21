# 12. AlexNet

## Learning Objectives

- Describe AlexNet's architecture and identify what changed relative to LeNet
- Explain why AlexNet's 2012 ImageNet win is considered the start of the deep learning era
- Implement AlexNet in PyTorch and understand each of its key innovations individually

## The Problem

Between LeNet (1998) and 2012, CNNs were known to work in principle but hadn't been shown to scale to large, complex, real-world image datasets — ImageNet, with over a million images across 1,000 categories, was widely seen as too hard for any existing method to meaningfully crack. AlexNet (Krizhevsky, Sutskever, Hinton, 2012) won the ImageNet competition that year by a massive margin (top-5 error of 15.3% vs the runner-up's 26.2%), and the deep learning era, as it's commonly dated, effectively starts here.

## The Concept

### AlexNet architecture

```
Input: 224x224x3
    ↓
Conv(96 filters, 11x11, stride 4)   → 55x55x96
    ↓ ReLU, MaxPool(3x3, stride 2)   → 27x27x96
Conv(256 filters, 5x5, same)        → 27x27x256
    ↓ ReLU, MaxPool(3x3, stride 2)   → 13x13x256
Conv(384 filters, 3x3, same)        → 13x13x384
    ↓ ReLU
Conv(384 filters, 3x3, same)        → 13x13x384
    ↓ ReLU
Conv(256 filters, 3x3, same)        → 13x13x256
    ↓ ReLU, MaxPool(3x3, stride 2)   → 6x6x256
Flatten                              → 9216
    ↓
FC(4096) → ReLU → Dropout
FC(4096) → ReLU → Dropout
FC(1000)                             → 1000 class scores (ImageNet classes)
```

Eight learnable layers total (5 conv + 3 FC) — roughly four times deeper than LeNet's two convolutional layers, made possible by two things that weren't yet standard when LeNet was designed: enough compute (via GPUs, used for training for the first time at this scale) and enough data (ImageNet's over a million labeled images, vs the tens of thousands LeNet trained on).

### What AlexNet introduced (or popularized)

**ReLU instead of tanh/sigmoid** (Lesson 06). AlexNet's paper directly credits ReLU with dramatically faster training — their reported figure was roughly 6x faster convergence to the same training error compared to tanh, on this specific network and dataset. This was the result that established ReLU as the default CNN activation, a choice that has held for over a decade since.

**Dropout** (Lesson 10). Applied to the FC layers, dropout was a critical part of preventing the massive number of FC parameters (over 58 million, in the two 4096-unit layers combined) from overfitting a "mere" million-image dataset.

**Training on GPUs**. AlexNet was originally split across two GPUs (a hardware limitation of the time, each GPU holding roughly half the network's filters, with limited cross-communication between them), and its training took about a week on the hardware available in 2012. This was one of the first large-scale demonstrations that GPU-parallelized training made deep networks practically trainable, a fact that shaped essentially all deep learning infrastructure that followed.

**Large first-layer filters (11×11, stride 4)**. Unlike the smaller 5×5 filters of LeNet or the even smaller 3×3 filters later architectures would standardize on (Lesson 13), AlexNet's first layer used large filters with a large stride, aggressively downsampling early to control the computational cost of processing 224×224 input. Later work (starting with VGG) found that replacing one large filter with several stacked small filters is usually both cheaper and more effective — a lesson the field learned specifically by moving past AlexNet's original design choice here.

**Local Response Normalization (LRN)**. AlexNet included a normalization scheme (competing activations across nearby channels) that the paper credited with a modest accuracy improvement. In practice, LRN was found in later work to contribute little and has been essentially abandoned in favor of batch normalization (Lesson 10), which wasn't invented until 2015. Most modern reimplementations of AlexNet drop LRN entirely, as this lesson's code does.

**Data augmentation** (Lesson 18) — random crops and horizontal flips of training images, plus a color-jittering technique (PCA-based color augmentation) — was used to artificially expand the effective size of the training set and reduce overfitting, foreshadowing augmentation's role as a near-universal technique in later architectures.

### Why this mattered beyond the architecture itself

AlexNet's win didn't just produce a better model — it changed what the field believed was possible, triggering a rapid wave of follow-up architectures (Lessons 13-16) each pushing depth, efficiency, or accuracy further. Every architecture in this module from here forward is a direct response, in some way, to what AlexNet demonstrated was achievable.

See `code/alexnet_demo.py` for a PyTorch implementation (LRN omitted, following modern convention), a layer-by-layer shape trace on 224×224×3 input, and a parameter count breakdown showing where AlexNet's ~60 million parameters live.

## Exercises

1. Trace AlexNet's output shape at every layer by hand for 224×224×3 input, then verify with the PyTorch implementation.
2. Compute the parameter count for the first conv layer (11×11, stride 4, 96 filters) vs the two 4096-unit FC layers combined. Confirm the FC layers dominate the total parameter count, similar to what Lesson 07 showed for a simpler network.
3. Replace AlexNet's 11×11 first-layer filter with a stack of smaller filters achieving a similar receptive field (this is exactly VGG's idea, previewed here — see Lesson 13) and compare parameter counts.
4. Load `torchvision.models.alexnet(weights="IMAGENET1K_V1")` and run inference on a sample image, printing the top-5 predicted classes.

## Key Terms

| Term | What it actually means |
|---|---|
| AlexNet | The 2012 CNN (Krizhevsky, Sutskever, Hinton) that won ImageNet by a large margin, widely credited with starting the deep learning era |
| Top-5 error | A classification metric counting a prediction correct if the true label is among the model's 5 highest-scoring classes; standard for large-scale ImageNet evaluation |
| Local Response Normalization (LRN) | A normalization scheme used in AlexNet, largely abandoned in later architectures in favor of batch normalization |
| GPU-parallelized training | Splitting or accelerating neural network training across graphics processing units, which AlexNet's scale first demonstrated was practically necessary and effective |
