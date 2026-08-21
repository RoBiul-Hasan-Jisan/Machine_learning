"""
Transfer learning with torchvision's ResNet-18: feature extraction
(frozen backbone) vs fine-tuning (unfrozen backbone), with trainable
parameter counts for each strategy and a two-phase training pattern.

Note: this script tries to download real ImageNet-pretrained weights.
If network access is unavailable, it falls back to a randomly
initialized ResNet-18 with the same architecture, so the *pattern*
(freezing, replacing the head, counting trainable params) still runs
and teaches correctly -- only the actual pretrained accuracy benefit
requires real downloaded weights.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def load_resnet18(pretrained=True):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if pretrained:
            print("Loaded ImageNet-pretrained ResNet-18 weights.")
        else:
            print("Loaded randomly initialized ResNet-18 (pretrained=False).")
        return model
    except Exception as e:
        print(f"Could not download pretrained weights ({type(e).__name__}: {e}).")
        print("Falling back to a randomly initialized ResNet-18 with the same")
        print("architecture -- the transfer-learning PATTERN below is identical")
        print("regardless of which weights are loaded.\n")
        return models.resnet18(weights=None)


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def feature_extraction_setup(num_new_classes=5):
    model = load_resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False  # freeze every pretrained layer

    model.fc = nn.Linear(model.fc.in_features, num_new_classes)  # new, trainable head
    return model


def fine_tuning_setup(num_new_classes=5, freeze_until_stage=None):
    """freeze_until_stage: None = fine-tune everything.
    Otherwise, one of 'layer1','layer2','layer3','layer4' - layers before
    this stage stay frozen, this stage onward (+ new head) are trainable."""
    model = load_resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_new_classes)

    if freeze_until_stage is None:
        return model  # everything trainable (all params default to requires_grad=True)

    stage_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    freeze = True
    for name, child in model.named_children():
        if name == freeze_until_stage:
            freeze = False
        if name == "fc":
            freeze = False  # always train the new head
        for param in child.parameters():
            param.requires_grad = not freeze
    return model


def demo_parameter_counts():
    print("=== Feature extraction (frozen backbone) ===")
    model_extract = feature_extraction_setup(num_new_classes=5)
    trainable, total = count_trainable_params(model_extract)
    print(f"Trainable: {trainable:,} / Total: {total:,}  ({trainable / total:.2%} trainable)\n")

    print("=== Full fine-tuning (everything unfrozen) ===")
    model_finetune = fine_tuning_setup(num_new_classes=5, freeze_until_stage=None)
    trainable, total = count_trainable_params(model_finetune)
    print(f"Trainable: {trainable:,} / Total: {total:,}  ({trainable / total:.2%} trainable)\n")

    print("=== Partial fine-tuning (only layer4 onward + head unfrozen) ===")
    model_partial = fine_tuning_setup(num_new_classes=5, freeze_until_stage="layer4")
    trainable, total = count_trainable_params(model_partial)
    print(f"Trainable: {trainable:,} / Total: {total:,}  ({trainable / total:.2%} trainable)\n")


def demo_two_phase_training():
    """Phase 1: train only the new head (frozen backbone).
    Phase 2: unfreeze everything and continue with a smaller LR."""
    torch.manual_seed(0)
    num_classes = 3
    X = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, num_classes, (16,))

    model = feature_extraction_setup(num_new_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()

    print("=== Phase 1: train new head only (frozen backbone) ===")
    optimizer_phase1 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    for epoch in range(3):
        optimizer_phase1.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer_phase1.step()
        print(f"Phase 1, epoch {epoch + 1}: loss={loss.item():.4f}")

    print("\n=== Phase 2: unfreeze everything, continue with a smaller LR ===")
    for param in model.parameters():
        param.requires_grad = True
    optimizer_phase2 = torch.optim.Adam(model.parameters(), lr=1e-5)  # smaller LR
    for epoch in range(3):
        optimizer_phase2.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer_phase2.step()
        print(f"Phase 2, epoch {epoch + 1}: loss={loss.item():.4f}")


if __name__ == "__main__":
    demo_parameter_counts()
    demo_two_phase_training()
