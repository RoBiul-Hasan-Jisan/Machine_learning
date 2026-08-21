"""
Depthwise separable convolution from scratch (via grouped conv), a
parameter-count comparison against a standard convolution, a compound
scaling illustration, and a simplified MBConv-style block.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels,  # groups=in_channels -> one filter per channel
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def demo_shape_and_param_comparison():
    in_c, out_c = 256, 256
    x = torch.randn(1, in_c, 14, 14)

    standard_conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    depthwise_sep = DepthwiseSeparableConv(in_c, out_c, kernel_size=3, stride=1)

    out_standard = standard_conv(x)
    out_dsep = depthwise_sep(x)

    print("Input shape:                    ", tuple(x.shape))
    print("Standard conv output shape:     ", tuple(out_standard.shape))
    print("Depthwise separable output shape:", tuple(out_dsep.shape))
    assert out_standard.shape == out_dsep.shape
    print("Same output shape, different cost.\n")

    standard_params = sum(p.numel() for p in standard_conv.parameters())
    dsep_params = sum(p.numel() for p in depthwise_sep.parameters())

    print(f"Standard conv params:            {standard_params:,}")
    print(f"Depthwise separable conv params: {dsep_params:,}")
    print(f"Reduction: {standard_params / dsep_params:.1f}x fewer parameters\n")

    # Match the lesson's hand-computed formula
    formula_standard = 3 * 3 * in_c * out_c
    formula_dsep = 3 * 3 * in_c + in_c * out_c
    print(f"Formula check - standard:  {formula_standard:,} (matches actual weight count, excluding biases)")
    print(f"Formula check - dsep:      {formula_dsep:,}\n")


def demo_compound_scaling():
    alpha, beta, gamma = 1.2, 1.1, 1.15
    print("phi | depth (d=alpha^phi) | width (w=beta^phi) | resolution (r=gamma^phi)")
    for phi in [0, 1, 2, 3]:
        d = alpha ** phi
        w = beta ** phi
        r = gamma ** phi
        print(f"{phi:3d} | {d:19.3f} | {w:18.3f} | {r:23.3f}")
    print("\nAs phi increases, all three dimensions grow together in a fixed ratio,")
    print("rather than scaling just one dimension in isolation.\n")


class SimplifiedMBConv(nn.Module):
    """expand (1x1) -> depthwise (3x3) -> project (1x1), with a skip
    connection when input/output shapes match."""

    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_skip = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,
                      stride=stride, padding=1, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_skip:
            out = out + x
        return out


def demo_mbconv_block():
    x = torch.randn(1, 32, 28, 28)

    block_with_skip = SimplifiedMBConv(32, 32, expand_ratio=4, stride=1)
    out_skip = block_with_skip(x)
    print("MBConv, same in/out channels & stride 1 (skip connection used):")
    print("Output shape:", tuple(out_skip.shape), "\n")

    block_downsample = SimplifiedMBConv(32, 64, expand_ratio=4, stride=2)
    out_downsample = block_downsample(x)
    print("MBConv, channel change & stride 2 (no skip connection):")
    print("Output shape:", tuple(out_downsample.shape))

    params = sum(p.numel() for p in block_with_skip.parameters())
    print(f"\nParameter count for one MBConv block (32->32, expand=4): {params:,}")


if __name__ == "__main__":
    print("=== Depthwise separable vs standard convolution ===")
    demo_shape_and_param_comparison()

    print("=== Compound scaling illustration ===")
    demo_compound_scaling()

    print("=== Simplified MBConv block ===")
    demo_mbconv_block()
