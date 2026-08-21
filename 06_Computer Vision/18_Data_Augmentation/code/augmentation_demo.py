"""
Standard image augmentations applied to a synthetic image, a demo of a
label-breaking augmentation (large rotation on a digit-like shape), and
a small experiment showing augmentation's effect on overfitting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def make_synthetic_photo(size=64):
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    img[10:30, 10:40] = [200, 60, 60]
    img += rng.integers(0, 10, size=img.shape, dtype=np.uint8)
    return Image.fromarray(img)


def demo_standard_augmentations():
    img = make_synthetic_photo()

    flip = transforms.RandomHorizontalFlip(p=1.0)(img)
    rotate = transforms.RandomRotation(degrees=10)(img)
    jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)(img)
    crop = transforms.RandomResizedCrop(64, scale=(0.6, 1.0))(img)

    print("Original image size:", img.size)
    print("Flipped image size:  ", flip.size)
    print("Rotated image size:  ", rotate.size)
    print("Jittered image size: ", jitter.size)
    print("Cropped image size:  ", crop.size)

    orig_arr = np.array(img).astype(float)
    flip_arr = np.array(flip).astype(float)
    diff = np.abs(orig_arr - flip_arr).mean()
    print("\nPixel difference (original vs flipped): {:.2f}".format(diff))
    print("(Nonzero, as expected -- flip changes pixel arrangement but not the label.)\n")


def demo_label_breaking_rotation():
    size = 40
    base = Image.new("L", (size, size), color=0)
    arr = np.array(base).copy()

    yy, xx = np.mgrid[0:size, 0:size]
    circle = (xx - 20) ** 2 + (yy - 28) ** 2 < 64
    tail = (xx > 12) & (xx < 20) & (yy > 5) & (yy < 25)
    arr[circle | tail] = 255
    img = Image.fromarray(arr)

    for angle in [0, 10, 90, 180]:
        rotated = img.rotate(angle)
        rotated_arr = np.array(rotated)
        ys, xs = np.nonzero(rotated_arr)
        com_y = ys.mean() if len(ys) else 0
        print("Rotation {:3d} deg -> shape center of mass (row): {:.1f}".format(angle, com_y))

    print("\nAt 180 degrees, the tail (originally pointing up-left near the top)")
    print("now points down-right near the bottom -- a shape that looked like '6'")
    print("now has the structural signature closer to '9'. This is exactly why")
    print("large rotations are unsafe augmentations for digit classification.\n")


class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def make_tiny_dataset(n_train=12, n_val=40, seed=0):
    rng = np.random.default_rng(seed)

    def make_batch(n, orientation, jitter=True):
        imgs = np.zeros((n, 1, 16, 16), dtype=np.float32)
        for i in range(n):
            if orientation == "vertical":
                col = rng.integers(6, 10) if jitter else 8
                imgs[i, 0, :, col - 1:col + 1] = 1.0
            else:
                row = rng.integers(6, 10) if jitter else 8
                imgs[i, 0, row - 1:row + 1, :] = 1.0
        return imgs

    X_train = np.concatenate([make_batch(n_train // 2, "vertical", jitter=False),
                               make_batch(n_train // 2, "horizontal", jitter=False)])
    y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

    X_val = np.concatenate([make_batch(n_val // 2, "vertical", jitter=True),
                             make_batch(n_val // 2, "horizontal", jitter=True)])
    y_val = np.array([0] * (n_val // 2) + [1] * (n_val // 2))

    return (torch.from_numpy(X_train), torch.from_numpy(y_train),
            torch.from_numpy(X_val), torch.from_numpy(y_val))


def augment_batch(X, max_shift=3):
    X_aug = X.clone()
    for i in range(X.shape[0]):
        shift_h = np.random.randint(-max_shift, max_shift + 1)
        shift_w = np.random.randint(-max_shift, max_shift + 1)
        X_aug[i, 0] = torch.roll(X_aug[i, 0], shifts=(shift_h, shift_w), dims=(0, 1))
    return X_aug


def train_and_eval(use_augmentation, n_epochs=60):
    torch.manual_seed(0)
    np.random.seed(0)
    X_train, y_train, X_val, y_val = make_tiny_dataset()

    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        X_batch = augment_batch(X_train) if use_augmentation else X_train
        logits = model(X_batch)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

    return train_acc, val_acc


def demo_overfitting_comparison():
    train_acc_no_aug, val_acc_no_aug = train_and_eval(use_augmentation=False)
    train_acc_aug, val_acc_aug = train_and_eval(use_augmentation=True)

    print("Without augmentation:")
    print("  Train accuracy: {:.3f}   Val accuracy: {:.3f}   (gap: {:.3f})".format(
        train_acc_no_aug, val_acc_no_aug, train_acc_no_aug - val_acc_no_aug))
    print("With augmentation (random shifts each epoch):")
    print("  Train accuracy: {:.3f}   Val accuracy: {:.3f}   (gap: {:.3f})".format(
        train_acc_aug, val_acc_aug, train_acc_aug - val_acc_aug))
    print("\n(A smaller train/val gap with augmentation indicates the model")
    print("relied less on memorizing exact pixel positions in the tiny training set.)")


if __name__ == "__main__":
    print("=== Standard augmentations ===")
    demo_standard_augmentations()

    print("=== Label-breaking rotation demo ===")
    demo_label_breaking_rotation()

    print("=== Augmentation's effect on overfitting a tiny dataset ===")
    demo_overfitting_comparison()
