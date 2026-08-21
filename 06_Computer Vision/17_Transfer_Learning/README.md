# 17. Transfer Learning

## Learning Objectives

- Explain why features learned on a large dataset transfer to smaller, related tasks
- Implement feature extraction (frozen backbone) and fine-tuning (unfrozen backbone) with a pretrained CNN
- Choose between these two strategies based on dataset size and similarity to the pretraining domain

## The Problem

Training a CNN like ResNet or EfficientNet from scratch (Lessons 11-16) requires a large labeled dataset (ImageNet has 1.4 million images) and substantial compute (days on capable GPUs, historically weeks). Most real projects don't have either — a few thousand labeled images and a laptop or a single cloud GPU is far more typical. Transfer learning solves this by reusing a network already trained on a large dataset, adapting it to a new, smaller task instead of starting from random weights.

## The Concept

### Why pretrained features transfer

Recall from Lesson 03: a CNN's early layers learn fairly generic features — edges, colors, textures — regardless of the specific classification task, while later layers learn increasingly task-specific combinations of those features. A network trained on ImageNet's 1,000 diverse categories has, in its early and middle layers, already learned a rich, general-purpose visual vocabulary that is highly likely to be useful for a *different* image classification task too, even one ImageNet was never trained on directly (medical images, satellite images, product photos) — as long as the new task's images are still, broadly, natural photographic images rather than something structurally very different (like raw audio spectrograms).

### Two transfer learning strategies

**Feature extraction (frozen backbone).** Keep every pretrained layer's weights fixed ("frozen" — no gradient updates), remove the original final classification layer, and train only a new, small classifier head on top of the frozen features.

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False        # freeze every pretrained layer

model.fc = nn.Linear(model.fc.in_features, num_new_classes)  # new, trainable head
# only model.fc's parameters have requires_grad=True now
```

This is fast to train (only the small new head has gradients to compute) and works well with a small new dataset, since there are far fewer parameters at risk of overfitting.

**Fine-tuning (unfrozen backbone).** Start from pretrained weights, but allow some or all of the backbone's layers to keep updating during training on the new dataset, typically with a much smaller learning rate than you'd use training from scratch (to avoid destroying the useful pretrained features with large, disruptive updates).

```python
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_new_classes)

# unfreeze everything, but use a small learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # note: much smaller than a from-scratch lr like 1e-3
```

A common middle-ground: freeze the early layers (most generic, least task-specific) and fine-tune only the later layers (more task-specific, most likely to need adjustment for the new task) plus the new head — balancing the risk of overfitting with limited data against giving the network room to adapt its more specialized features.

### Choosing a strategy

| Dataset size | Similarity to pretraining domain | Recommended strategy |
|---|---|---|
| Small | Similar (e.g. new set of natural photos) | Feature extraction (frozen backbone) — little data, risk of overfitting if you unfreeze too much |
| Small | Different (e.g. medical/satellite imagery) | Fine-tune just the last few layers — some adaptation needed, but still guard against overfitting |
| Large | Similar | Fine-tune the whole network — enough data to safely adjust everything |
| Large | Different | Fine-tune the whole network, possibly for longer, or consider training from scratch if the domain gap is severe |

The intuition scales with how much labeled data you have (more data → more layers you can safely unfreeze without overfitting) and how different the new domain is from natural photographic images (more different → more layers likely need adjusting, since even "generic" edge/texture features may need to shift).

### Preprocessing must match the pretrained model's expectations

Pretrained weights were learned assuming a specific input preprocessing — a specific input resolution and specific per-channel normalization statistics (typically ImageNet's channel means and standard deviations). Using different preprocessing at inference time silently degrades performance, since the pretrained filters were tuned to expect inputs in a particular numeric range and distribution:

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet channel means
                          std=[0.229, 0.224, 0.225]),    # ImageNet channel stds
])
```

### A practical training pattern

A common, effective recipe: start with feature extraction (fast, safe) to get a reasonable baseline quickly, then switch to fine-tuning (unfreeze some or all layers, drop the learning rate) for a further accuracy boost once the new head has already learned something sensible on top of the frozen features — unfreezing everything from the start, with randomly initialized head weights still producing large, noisy gradients, risks disrupting the pretrained backbone before the new head has stabilized.

See `code/transfer_learning_demo.py` for both strategies implemented on `torchvision`'s pretrained ResNet-18, including a parameter count showing exactly how many parameters are trainable in each case, and a demonstration of the two-phase (extract-then-fine-tune) pattern.

## Exercises

1. Load a pretrained ResNet-18, freeze all layers, replace the final FC layer, and print the total vs trainable parameter count. Confirm only the new head's parameters are trainable.
2. Repeat with a fine-tuning setup that unfreezes only the last residual stage (Lesson 15) plus the new head, and compare the trainable parameter count to full fine-tuning.
3. Train a frozen-backbone classifier and a fully fine-tuned classifier on the same small synthetic dataset for the same number of epochs. Compare training speed and final accuracy.
4. Implement the two-phase pattern: train the new head with a frozen backbone for a few epochs, then unfreeze the whole network and continue training with a smaller learning rate. Compare the result to fine-tuning from the start with all layers unfrozen.

## Key Terms

| Term | What it actually means |
|---|---|
| Transfer learning | Reusing a network pretrained on one (typically large) dataset as a starting point for a different, usually smaller, task |
| Feature extraction (frozen backbone) | Transfer learning where pretrained layers are kept fixed and only a new classifier head is trained |
| Fine-tuning | Transfer learning where some or all pretrained layers continue to update during training on the new task, typically with a small learning rate |
| Backbone | The main feature-extracting portion of a CNN (everything before the final classification layer), often reused across tasks |
| Domain gap | The degree of difference between the pretraining dataset's images and the new task's images, which influences how much fine-tuning is needed |
