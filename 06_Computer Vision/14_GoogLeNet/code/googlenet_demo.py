"""
Inception module (with 1x1 bottleneck), a cost comparison vs the naive
version, and a simplified GoogLeNet-style network stacking several
Inception modules with global average pooling.
"""

import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # concatenate along channel dim


class SimplifiedGoogLeNet(nn.Module):
    """A small stack of Inception modules with global average pooling,
    illustrating the pattern without reproducing all 9 modules of the
    original 22-layer network."""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception1 = InceptionModule(64, out_1x1=32, reduce_3x3=48, out_3x3=64,
                                           reduce_5x5=8, out_5x5=16, pool_proj=16)
        # inception1 output channels: 32 + 64 + 16 + 16 = 128
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception2 = InceptionModule(128, out_1x1=64, reduce_3x3=64, out_3x3=96,
                                           reduce_5x5=16, out_5x5=32, pool_proj=32)
        # inception2 output channels: 64 + 96 + 32 + 32 = 224

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(224, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.pool(x)
        x = self.inception2(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.fc(x)


def demo_inception_branch_shapes():
    module = InceptionModule(in_channels=256, out_1x1=64, reduce_3x3=96, out_3x3=128,
                              reduce_5x5=16, out_5x5=32, pool_proj=32)
    x = torch.randn(1, 256, 28, 28)

    b1 = module.branch1(x)
    b2 = module.branch2(x)
    b3 = module.branch3(x)
    b4 = module.branch4(x)

    print("Input shape:          ", tuple(x.shape))
    print("1x1 branch output:    ", tuple(b1.shape))
    print("3x3 branch output:    ", tuple(b2.shape))
    print("5x5 branch output:    ", tuple(b3.shape))
    print("pool branch output:   ", tuple(b4.shape))

    assert b1.shape[2:] == b2.shape[2:] == b3.shape[2:] == b4.shape[2:]
    print("All branches share the same spatial size -> safe to concatenate.\n")

    out = module(x)
    print("Concatenated output shape:", tuple(out.shape))
    assert out.shape[1] == 64 + 128 + 32 + 32
    print(f"Channel count: {out.shape[1]} = 64 + 128 + 32 + 32\n")


def demo_bottleneck_cost_comparison():
    in_c, out_c, reduce_c = 256, 32, 16

    params_without_bottleneck = 5 * 5 * in_c * out_c
    params_with_bottleneck = (1 * 1 * in_c * reduce_c) + (5 * 5 * reduce_c * out_c)

    print(f"5x5 branch, {in_c} -> {out_c} channels:")
    print(f"  WITHOUT 1x1 bottleneck: {params_without_bottleneck:,} params")
    print(f"  WITH 1x1 bottleneck ({in_c}->{reduce_c}->{out_c}):    {params_with_bottleneck:,} params")
    print(f"  Reduction: {params_without_bottleneck / params_with_bottleneck:.1f}x fewer parameters\n")


def demo_simplified_googlenet():
    model = SimplifiedGoogLeNet(num_classes=10)
    x = torch.randn(1, 3, 224, 224)

    print("=== Simplified GoogLeNet-style network ===")
    h = model.stem(x)
    print(f"After stem:        {tuple(h.shape)}")
    h = model.inception1(h)
    print(f"After inception1:  {tuple(h.shape)}")
    h = model.pool(h)
    print(f"After pool:        {tuple(h.shape)}")
    h = model.inception2(h)
    print(f"After inception2:  {tuple(h.shape)}")
    h = model.global_avg_pool(h)
    print(f"After global avg pool: {tuple(h.shape)}")

    out = model(x)
    assert out.shape == (1, 10)
    print(f"Final output shape: {tuple(out.shape)}\n")

    total_params = sum(p.numel() for p in model.parameters())
    fc_params = sum(p.numel() for name, p in model.named_parameters() if "fc" in name)
    print(f"Total parameters: {total_params:,}")
    print(f"FC layer parameters: {fc_params:,} ({fc_params / total_params:.1%} of total)")
    print("(Global average pooling keeps the final FC layer tiny, unlike")
    print(" AlexNet/VGG's large flatten+FC classifiers.)")


if __name__ == "__main__":
    print("=== Inception module branch shapes ===")
    demo_inception_branch_shapes()

    print("=== 1x1 bottleneck cost comparison ===")
    demo_bottleneck_cost_comparison()

    demo_simplified_googlenet()
