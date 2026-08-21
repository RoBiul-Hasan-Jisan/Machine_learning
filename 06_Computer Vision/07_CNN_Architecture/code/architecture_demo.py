"""
A full CNN assembled from Conv/Activation/Pool/FC blocks, with
layer-by-layer shape verification and a parameter count breakdown
matching the formulas from the lesson.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # same padding
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(56 * 56 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, verbose=False):
        if verbose:
            print(f"Input:                {tuple(x.shape)}")
        x = torch.relu(self.conv1(x))
        if verbose:
            print(f"After conv1 + ReLU:   {tuple(x.shape)}")
        x = self.pool1(x)
        if verbose:
            print(f"After pool1:          {tuple(x.shape)}")
        x = torch.relu(self.conv2(x))
        if verbose:
            print(f"After conv2 + ReLU:   {tuple(x.shape)}")
        x = self.pool2(x)
        if verbose:
            print(f"After pool2:          {tuple(x.shape)}")
        x = x.flatten(1)
        if verbose:
            print(f"After flatten:        {tuple(x.shape)}")
        x = torch.relu(self.fc1(x))
        if verbose:
            print(f"After fc1 + ReLU:     {tuple(x.shape)}")
        x = self.fc2(x)
        if verbose:
            print(f"After fc2 (output):   {tuple(x.shape)}")
        return x


def conv_param_count(f, c_in, c_out):
    return (f * f * c_in + 1) * c_out


def fc_param_count(in_size, out_size):
    return (in_size + 1) * out_size


def demo_shape_trace():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    print("=== Layer-by-layer shape trace ===")
    out = model(x, verbose=True)
    assert out.shape == (1, 10)
    print("\nFinal output shape matches expected (1, 10).\n")


def demo_param_counts():
    print("=== Parameter count breakdown ===")
    conv1_params = conv_param_count(3, 3, 64)
    conv2_params = conv_param_count(3, 64, 128)
    fc1_params = fc_param_count(56 * 56 * 128, 256)
    fc2_params = fc_param_count(256, 10)

    print(f"conv1 (3x3, 3->64):     {conv1_params:,}")
    print(f"conv2 (3x3, 64->128):   {conv2_params:,}")
    print(f"fc1 (401408 -> 256):    {fc1_params:,}")
    print(f"fc2 (256 -> 10):        {fc2_params:,}")
    print(f"Total (hand-computed):  {conv1_params + conv2_params + fc1_params + fc2_params:,}")

    model = SimpleCNN(num_classes=10)
    torch_total = sum(p.numel() for p in model.parameters())
    print(f"Total (torch actual):   {torch_total:,}")

    assert torch_total == conv1_params + conv2_params + fc1_params + fc2_params
    print("\nHand-computed parameter counts match torch exactly.")
    print(f"\nNote: fc1 alone accounts for {fc1_params / torch_total:.1%} of all parameters —")
    print("this is exactly why architectures favor global average pooling at scale.")


if __name__ == "__main__":
    demo_shape_trace()
    demo_param_counts()
