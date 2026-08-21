"""
VGG-16 implemented from a configurable block structure, plus a
receptive-field / parameter comparison between one 5x5 layer and
two stacked 3x3 layers.
"""

import torch
import torch.nn as nn

VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def build_vgg_features(cfg, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, cfg=VGG16_CFG, num_classes=1000):
        super().__init__()
        self.features = build_vgg_features(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def demo_receptive_field_comparison():
    """Compare one 5x5 conv layer vs two stacked 3x3 conv layers:
    same effective receptive field / output size, fewer parameters."""
    x = torch.randn(1, 1, 9, 9)

    single_5x5 = nn.Conv2d(1, 1, kernel_size=5)
    out_single = single_5x5(x)

    stacked_3x3 = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3),
        nn.Conv2d(1, 1, kernel_size=3),
    )
    out_stacked = stacked_3x3(x)

    print("Input shape:                      ", tuple(x.shape))
    print("Output shape after one 5x5 conv:  ", tuple(out_single.shape))
    print("Output shape after two 3x3 convs: ", tuple(out_stacked.shape))
    assert out_single.shape == out_stacked.shape
    print("Confirmed: same output size -> same effective receptive field.\n")

    c = 256  # example channel count, as used in the lesson
    params_5x5 = 5 * 5 * c * c
    params_3x3_stacked = 2 * (3 * 3 * c * c)
    savings = 1 - params_3x3_stacked / params_5x5
    print(f"Parameters for one 5x5 layer ({c}->{c} channels):        {params_5x5:,}")
    print(f"Parameters for two stacked 3x3 layers ({c}->{c}):        {params_3x3_stacked:,}")
    print(f"Savings: {savings:.1%}\n")

    params_7x7 = 7 * 7 * c * c
    params_3x3_x3 = 3 * (3 * 3 * c * c)
    savings_7 = 1 - params_3x3_x3 / params_7x7
    print(f"Parameters for one 7x7 layer ({c}->{c} channels):        {params_7x7:,}")
    print(f"Parameters for three stacked 3x3 layers ({c}->{c}):      {params_3x3_x3:,}")
    print(f"Savings: {savings_7:.1%}\n")


def demo_vgg16_shape_and_params():
    model = VGG(VGG16_CFG, num_classes=1000)
    x = torch.randn(1, 3, 224, 224)

    print("=== VGG-16 shape trace (block outputs only) ===")
    h = x
    block_num = 1
    channels_seen = None
    for layer in model.features:
        h = layer(h)
        if isinstance(layer, nn.MaxPool2d):
            print(f"After block {block_num} (+ pool): {tuple(h.shape)}")
            block_num += 1

    out = model(x)
    assert out.shape == (1, 1000)
    print(f"\nFinal output shape: {tuple(out.shape)}\n")

    total_params = sum(p.numel() for p in model.parameters())
    conv_params = sum(p.numel() for name, p in model.named_parameters() if "features" in name)
    fc_params = sum(p.numel() for name, p in model.named_parameters() if "classifier" in name)

    print("=== Parameter breakdown ===")
    print(f"Conv layers total:  {conv_params:,}  ({conv_params / total_params:.1%})")
    print(f"FC layers total:    {fc_params:,}  ({fc_params / total_params:.1%})")
    print(f"Total:              {total_params:,}")
    print("(Matches the well-known ~138 million parameter count for VGG-16.)")


if __name__ == "__main__":
    print("=== Receptive field / parameter comparison: stacked 3x3 vs larger filters ===")
    demo_receptive_field_comparison()

    print("=== VGG-16 full model ===")
    demo_vgg16_shape_and_params()
