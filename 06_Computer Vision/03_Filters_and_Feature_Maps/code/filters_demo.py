"""
Classic hand-designed filters applied to a synthetic image, plus a tiny
trained CNN whose learned first-layer filters can be inspected directly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_synthetic_image(size=20):
    """A synthetic image with a vertical bar, a horizontal bar, and a flat region."""
    img = np.zeros((size, size), dtype=np.float32)
    img[:, size // 2 - 1: size // 2 + 1] = 1.0   # vertical bar
    img[2:4, :] = 0.6                             # horizontal bar
    return img


def apply_filter(img, kernel):
    t_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    t_kernel = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.conv2d(t_img, t_kernel)
    return out.squeeze().numpy()


VERTICAL_EDGE = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

HORIZONTAL_EDGE = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
], dtype=np.float32)

BLUR = np.ones((3, 3), dtype=np.float32) / 9.0

SHARPEN = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
], dtype=np.float32)


def demo_hand_designed_filters():
    img = make_synthetic_image()

    v_response = apply_filter(img, VERTICAL_EDGE)
    h_response = apply_filter(img, HORIZONTAL_EDGE)

    print("Max |response| from vertical-edge filter:", round(np.abs(v_response).max(), 3))
    print("Max |response| from horizontal-edge filter on the same image:",
          round(np.abs(h_response).max(), 3))

    # The vertical bar should trigger the vertical-edge filter strongly
    # in the columns adjacent to the bar.
    col_with_bar = v_response[:, img.shape[1] // 2 - 2]
    flat_region = v_response[:, 2]
    print("\nVertical-edge response near the vertical bar (should be large):",
          round(np.abs(col_with_bar).max(), 3))
    print("Vertical-edge response in a flat region (should be near zero):",
          round(np.abs(flat_region).max(), 3))


def demo_blur_and_sharpen():
    rng = np.random.default_rng(0)
    noisy_img = make_synthetic_image() + rng.normal(0, 0.05, (20, 20)).astype(np.float32)

    blurred = apply_filter(noisy_img, BLUR)
    sharpened = apply_filter(noisy_img, SHARPEN)

    print("\nInput std:", round(noisy_img.std(), 4))
    print("Blurred std (should be lower - smoothing reduces variance):", round(blurred.std(), 4))
    print("Sharpened std (should be higher - amplifies local contrast):", round(sharpened.std(), 4))


class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 10 * 10, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def demo_learned_filters():
    """Train briefly on a toy 2-class task (bar orientation) and inspect
    the learned first-layer filters."""
    rng = np.random.default_rng(1)

    def make_batch(n, orientation):
        imgs = np.zeros((n, 1, 20, 20), dtype=np.float32)
        for i in range(n):
            if orientation == "vertical":
                col = rng.integers(4, 16)
                imgs[i, 0, :, col - 1:col + 1] = 1.0
            else:
                row = rng.integers(4, 16)
                imgs[i, 0, row - 1:row + 1, :] = 1.0
        return imgs

    X_vert = make_batch(64, "vertical")
    X_horiz = make_batch(64, "horizontal")
    X = np.concatenate([X_vert, X_horiz], axis=0)
    y = np.array([0] * 64 + [1] * 64)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(30):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    print(f"\nTiny CNN trained to distinguish vertical vs horizontal bars: accuracy = {acc:.3f}")

    learned_filters = model.conv1.weight.detach().numpy()  # shape (8, 1, 3, 3)
    print("Learned first-layer filters (8 filters, 3x3 each):")
    for i, f in enumerate(learned_filters):
        print(f"Filter {i}:\n{f[0].round(2)}")


if __name__ == "__main__":
    print("=== Hand-designed edge filters ===")
    demo_hand_designed_filters()

    print("\n=== Blur vs sharpen ===")
    demo_blur_and_sharpen()

    print("\n=== Learned filters from a tiny trained CNN ===")
    demo_learned_filters()
