"""
Residual block with shape-mismatch handling, a gradient-flow comparison
between a deep plain network and a deep residual network, and a small
ResNet-style network built by stacking blocks into stages.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class PlainBlock(nn.Module):
    """Same as ResidualBlock but WITHOUT the skip connection, for comparison."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)  # no += x here


def make_stage(block_cls, in_channels, out_channels, num_blocks, stride):
    layers = [block_cls(in_channels, out_channels, stride=stride)]
    for _ in range(num_blocks - 1):
        layers.append(block_cls(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, block_cls=ResidualBlock):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = make_stage(block_cls, 32, 32, num_blocks=2, stride=1)
        self.stage2 = make_stage(block_cls, 32, 64, num_blocks=2, stride=2)
        self.stage3 = make_stage(block_cls, 64, 128, num_blocks=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.fc(x)


def demo_block_shape_handling():
    x_same = torch.randn(1, 32, 16, 16)
    block_no_change = ResidualBlock(32, 32, stride=1)
    out_same = block_no_change(x_same)
    print("Same channels, stride 1 -> shortcut is Identity:", isinstance(block_no_change.shortcut, nn.Identity))
    print("Output shape:", tuple(out_same.shape), "\n")

    x_diff = torch.randn(1, 32, 16, 16)
    block_downsample = ResidualBlock(32, 64, stride=2)
    out_diff = block_downsample(x_diff)
    print("Channel change + stride 2 -> shortcut uses a 1x1 conv:", not isinstance(block_downsample.shortcut, nn.Identity))
    print("Output shape:", tuple(out_diff.shape), "\n")


def demo_gradient_flow_comparison(depth=30):
    """Compare gradient magnitude reaching the FIRST layer of a deep
    plain network vs a deep residual network."""
    torch.manual_seed(0)

    plain_layers = nn.Sequential(*[PlainBlock(16, 16, stride=1) for _ in range(depth)])
    resid_layers = nn.Sequential(*[ResidualBlock(16, 16, stride=1) for _ in range(depth)])

    x_plain = torch.randn(2, 16, 8, 8, requires_grad=True)
    x_resid = x_plain.clone().detach().requires_grad_(True)

    out_plain = plain_layers(x_plain)
    out_resid = resid_layers(x_resid)

    out_plain.sum().backward()
    out_resid.sum().backward()

    plain_grad_norm = x_plain.grad.norm().item()
    resid_grad_norm = x_resid.grad.norm().item()

    print(f"Gradient norm reaching the input, {depth}-block PLAIN network:    {plain_grad_norm:.6f}")
    print(f"Gradient norm reaching the input, {depth}-block RESIDUAL network: {resid_grad_norm:.6f}")
    print("(With random initialization, a deep plain network's gradient magnitude")
    print(" is highly unstable with depth -- it can explode OR vanish depending on")
    print(" initialization scale, since every block's Jacobian multiplies in. The")
    print(" residual network's skip connections guarantee a direct, well-scaled")
    print(" identity path for gradient, keeping magnitudes far more controlled.)\n")


def demo_small_resnet():
    model = SmallResNet(num_classes=10)
    x = torch.randn(1, 3, 32, 32)

    print("=== SmallResNet shape trace ===")
    h = model.stem(x)
    print(f"After stem:   {tuple(h.shape)}")
    h = model.stage1(h)
    print(f"After stage1: {tuple(h.shape)}")
    h = model.stage2(h)
    print(f"After stage2: {tuple(h.shape)}")
    h = model.stage3(h)
    print(f"After stage3: {tuple(h.shape)}")
    h = model.global_avg_pool(h)
    print(f"After global avg pool: {tuple(h.shape)}")

    out = model(x)
    assert out.shape == (1, 10)
    print(f"Final output shape: {tuple(out.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    print("=== Residual block shape handling ===")
    demo_block_shape_handling()

    print("=== Gradient flow: plain vs residual, deep stack ===")
    demo_gradient_flow_comparison()

    print("=== Small ResNet-style network ===")
    demo_small_resnet()
