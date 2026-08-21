# 18. Data Augmentation

## Learning Objectives

- Explain why data augmentation reduces overfitting without collecting new data
- Apply standard image augmentations (flips, crops, rotation, color jitter) and know when each is appropriate
- Distinguish augmentations that preserve labels from those that can silently break them

## The Problem

CNNs, especially deep ones (Lessons 12-16), have millions of parameters and can overfit a limited training set — memorizing specific images rather than learning generalizable visual patterns. Collecting more labeled data is the most direct fix, but is often expensive or slow. Data augmentation manufactures additional, varied training examples from the data you already have, by applying label-preserving transformations to existing images.

## The Concept

### Why augmentation works

A photo of a cat is still a photo of a cat if it's flipped horizontally, shifted a few pixels, slightly rotated, or has its brightness adjusted. A model that's only ever seen that cat in one exact pixel arrangement can latch onto incidental details (this exact cat is always in the top-left, this exact cat is always well-lit) rather than the actual visual features that generalize. Augmentation exposes the model to many plausible variations of the same underlying image during training, encouraging it to learn features robust to those variations — directly counteracting the tendency to memorize.

Critically, augmentation is normally applied **only at training time**, generating a different randomly-augmented version of each image every epoch (so the model effectively never sees the exact same input twice), while validation and test data are evaluated on the original, unaugmented images — augmenting test data would change what you're measuring.

### Standard image augmentations

| Augmentation | What it does | When to use it |
|---|---|---|
| Horizontal flip | Mirrors the image left-right | Safe for most natural images (a flipped cat is still a cat); NOT safe for tasks where left/right matters (text, handedness) |
| Random crop | Crops a random sub-region, then resizes back to the input size | Simulates variation in framing/zoom; very commonly used |
| Rotation (small angle) | Rotates the image by a few degrees | Safe for most natural photos; be cautious with large rotations (a heavily rotated digit "6" can look like a "9") |
| Color jitter | Randomly adjusts brightness, contrast, saturation | Simulates different lighting conditions and camera settings |
| Random erasing / cutout | Blacks out a random rectangular region | Forces the model to not rely on any single localized region, similar in spirit to dropout (Lesson 10) but applied to the input |
| Translation (shift) | Shifts the image by a few pixels in x/y | Simulates imperfect centering/framing |

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Lesson 17
])

val_transform = transforms.Compose([         # no random augmentation at eval time
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Augmentations that can silently break your labels

Not every plausible-sounding transformation actually preserves the label — this is a genuine risk, not just a theoretical caveat:

- **Horizontal flip on text or handedness-relevant images**: flipping an image containing readable text, a traffic sign with directional meaning, or a task where left/right orientation is the actual label (e.g. "which way is this arrow pointing") changes the correct answer.
- **Large rotation on digit/character recognition**: rotating "6" by 180 degrees produces something closer to "9" — the augmentation has silently created a mislabeled example.
- **Aggressive color jitter on tasks where color is the signal** (e.g. classifying ripeness of fruit by color, or medical imaging where color/intensity encodes diagnostic information): color augmentation can wash out the actual signal the model needs to learn.
- **Random cropping that can crop out the entire object of interest**: for object-centric tasks where the object might be small or off-center, an aggressive random crop can produce an image with no trace of the labeled object left, teaching the model an incorrect association between "empty background" and the original label.

The general rule: an augmentation is safe exactly when a human, shown the image before and after, would still assign the identical label. When in doubt, visualize augmented examples directly rather than assuming a standard augmentation "just works" for your specific task.

### AutoAugment and learned augmentation policies

Choosing augmentation types and strengths by hand is itself a hyperparameter search problem. **AutoAugment** (and related methods like RandAugment) search over augmentation policies automatically, using a validation set to find the combination and strength of augmentations that most improves generalization for a specific dataset, rather than relying on hand-picked defaults. These are available as drop-in `torchvision.transforms` options for common benchmark datasets and can be a fast way to get a well-tuned augmentation policy without manual experimentation.

### Mixup and CutMix: augmenting across images, not just within one

Two more advanced techniques blend *pairs* of training images (and their labels) rather than transforming a single image in isolation:

- **Mixup**: linearly blends two images' pixels and their labels by the same random weight (`lambda * image_A + (1-lambda) * image_B`, with the target label mixed the same way).
- **CutMix**: pastes a rectangular patch from one image onto another, mixing the label in proportion to the patch's area.

Both push the model toward smoother, more calibrated decision boundaries between classes and are common in modern training recipes for the architectures in Lessons 15-16, though they add complexity and are typically introduced only after the simpler augmentations above are already in place.

See `code/augmentation_demo.py` for a runnable comparison of standard augmentations applied to a synthetic image, a demonstration of a label-breaking augmentation (rotation flipping a "6"-like shape into something closer to a "9"-like shape), and a small experiment showing augmentation's effect on overfitting a tiny dataset.

## Exercises

1. Apply `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter` to the same image and visualize all three outputs side by side.
2. Construct a synthetic "6"-shaped image and rotate it by increasing angles (10°, 90°, 180°). At what rotation does it start to visually resemble a different digit? Explain why this makes large rotation unsafe for digit classification.
3. Train the same small CNN on a tiny dataset (e.g. 40 images) with and without augmentation for the same number of epochs. Compare the gap between training accuracy and validation accuracy in each case.
4. Implement Mixup from scratch (blend two images and their one-hot labels by a random `lambda`) and visualize a few mixed examples.

## Key Terms

| Term | What it actually means |
|---|---|
| Data augmentation | Generating additional training examples by applying label-preserving transformations to existing data |
| Label-preserving transformation | A transformation that does not change the correct label a human would assign to the resulting image |
| Random crop | An augmentation that crops a random sub-region of an image and resizes it back to the model's input size |
| AutoAugment / RandAugment | Methods that automatically search for effective augmentation policies rather than relying on hand-picked transformations |
| Mixup / CutMix | Augmentation techniques that blend pairs of images and their labels together, rather than transforming a single image alone |
