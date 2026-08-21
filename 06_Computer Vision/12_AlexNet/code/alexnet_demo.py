"""
AlexNet (LRN omitted, following modern convention) implemented in
PyTorch, with a layer-by-layer shape trace and parameter count breakdown.
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),      # 224 -> 54 (approx, matches paper's 55 with slight padding conventions)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),    # "same"-ish padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # ensures exactly 6x6 regardless of small input-size rounding
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


def demo_shape_trace():
    model = AlexNet(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)

    print("=== AlexNet layer-by-layer shape trace ===")
    print(f"Input:                    {tuple(x.shape)}")
    h = x
    for layer in model.features:
        h = layer(h)
        print(f"After {layer.__class__.__name__:12s}:     {tuple(h.shape)}")
    h = model.avgpool(h)
    print(f"After AdaptiveAvgPool2d:  {tuple(h.shape)}")
    h_flat = h.flatten(1)
    print(f"After Flatten:            {tuple(h_flat.shape)}")
    for layer in model.classifier:
        h_flat = layer(h_flat)
        print(f"After {layer.__class__.__name__:12s}:     {tuple(h_flat.shape)}")

    out = model(x)
    assert out.shape == (1, 1000)
    print("\nFinal output shape matches expected (1, 1000).\n")


def demo_param_breakdown():
    model = AlexNet(num_classes=1000)

    conv_params = sum(p.numel() for name, p in model.named_parameters() if "features" in name)
    fc_params = sum(p.numel() for name, p in model.named_parameters() if "classifier" in name)
    total = conv_params + fc_params

    print("=== Parameter breakdown ===")
    print(f"Conv layers total:  {conv_params:,}  ({conv_params / total:.1%} of total)")
    print(f"FC layers total:    {fc_params:,}  ({fc_params / total:.1%} of total)")
    print(f"Total:              {total:,}")
    print("\n(As in Lesson 07's SimpleCNN, the fully connected layers dominate")
    print(" the parameter count, even though the conv layers do most of the")
    print(" actual feature-extraction work.)\n")

    fc1_params = 256 * 6 * 6 * 4096 + 4096
    print(f"Just the first FC layer (9216 -> 4096): {fc1_params:,} params")
    print(f"That's {fc1_params / total:.1%} of AlexNet's entire parameter count, in one layer.")


if __name__ == "__main__":
    demo_shape_trace()
    demo_param_breakdown()
