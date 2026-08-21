"""
LeNet-5 (modern variant: ReLU + max pooling) implemented in PyTorch,
with a layer-by-layer shape trace, parameter count, and a short
training run on a synthetic digit-like dataset.
"""

import numpy as np
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),   # 32x32 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),  # 14x14 -> 10x10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def demo_shape_trace():
    model = LeNet5()
    x = torch.randn(1, 1, 32, 32)

    print("=== LeNet-5 layer-by-layer shape trace ===")
    print(f"Input:                    {tuple(x.shape)}")
    h = x
    for layer in model.features:
        h = layer(h)
        print(f"After {layer.__class__.__name__:12s}:     {tuple(h.shape)}")
    h_flat = h.flatten(1)
    print(f"After Flatten:            {tuple(h_flat.shape)}")
    for layer in model.classifier:
        h_flat = layer(h_flat)
        print(f"After {layer.__class__.__name__:12s}:     {tuple(h_flat.shape)}")

    out = model(x)
    assert out.shape == (1, 10)
    print("\nFinal output shape matches expected (1, 10).\n")


def demo_param_count():
    model = LeNet5()
    total = sum(p.numel() for p in model.parameters())
    print("=== Parameter count per layer ===")
    for name, param in model.named_parameters():
        print(f"{name:30s} {tuple(param.shape)}  ->  {param.numel():,} params")
    print(f"\nTotal parameters: {total:,}")
    print("(For comparison: Lesson 07's SimpleCNN had over 100 million, mostly")
    print(" in one FC layer. LeNet-5 is tiny by modern standards.)\n")


def make_toy_digit_dataset(n=200, seed=0):
    """Synthetic 'digit-like' task: distinguish a filled square (class 0)
    from a filled ring (class 1) on a 32x32 canvas."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, 32, 32), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    cy, cx = 16, 16
    yy, xx = np.mgrid[0:32, 0:32]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    for i in range(n):
        label = rng.integers(0, 2)
        y[i] = label
        if label == 0:
            X[i, 0] = (dist < 8).astype(np.float32)          # filled disk
        else:
            X[i, 0] = ((dist > 6) & (dist < 8)).astype(np.float32)  # ring
        X[i, 0] += rng.normal(0, 0.05, size=(32, 32)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def demo_training_run():
    X, y = make_toy_digit_dataset()
    model = LeNet5(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("=== Short training run on synthetic disk-vs-ring dataset ===")
    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0 or epoch == 9:
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            print(f"Epoch {epoch + 1:2d}: loss={loss.item():.4f}  accuracy={acc:.3f}")


if __name__ == "__main__":
    demo_shape_trace()
    demo_param_count()
    demo_training_run()
