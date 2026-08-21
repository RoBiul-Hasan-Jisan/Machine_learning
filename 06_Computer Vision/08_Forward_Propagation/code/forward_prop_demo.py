"""
Complete from-scratch forward pass for a small CNN:
Conv -> ReLU -> MaxPool(with argmax) -> Flatten -> FC -> Softmax,
verified against an equivalent PyTorch model with identical weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(x, filters, bias):
    """x: (H, W, C_in). filters: (K, f, f, C_in). bias: (K,). Returns (H_out, W_out, K)."""
    ih, iw, c_in = x.shape
    k, fh, fw, _ = filters.shape
    oh, ow = ih - fh + 1, iw - fw + 1

    output = np.zeros((oh, ow, k))
    for filt_idx in range(k):
        for i in range(oh):
            for j in range(ow):
                patch = x[i:i + fh, j:j + fw, :]
                output[i, j, filt_idx] = np.sum(patch * filters[filt_idx]) + bias[filt_idx]
    return output


def relu(x):
    return np.maximum(0, x)


def max_pool_with_argmax(x, f=2, s=2):
    """x: (H, W, C). Returns pooled (H_out, W_out, C) and argmax positions for backprop."""
    ih, iw, c = x.shape
    oh, ow = (ih - f) // s + 1, (iw - f) // s + 1
    pooled = np.zeros((oh, ow, c))
    argmax_positions = np.zeros((oh, ow, c, 2), dtype=int)  # (row, col) within the window

    for ch in range(c):
        for i in range(oh):
            for j in range(ow):
                row, col = i * s, j * s
                window = x[row:row + f, col:col + f, ch]
                flat_idx = np.argmax(window)
                local_row, local_col = np.unravel_index(flat_idx, window.shape)
                pooled[i, j, ch] = window[local_row, local_col]
                argmax_positions[i, j, ch] = [row + local_row, col + local_col]

    return pooled, argmax_positions


def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)


def forward_pass(x, filters, conv_bias, W2, b2):
    z1 = conv_layer(x, filters, conv_bias)
    a1 = relu(z1)
    p1, argmax_positions = max_pool_with_argmax(a1, f=2, s=2)
    f1 = p1.flatten()
    z2 = W2 @ f1 + b2
    y_hat = softmax(z2)
    cache = {"z1": z1, "a1": a1, "p1": p1, "argmax": argmax_positions, "f1": f1, "z2": z2}
    return y_hat, cache


def demo_softmax_matches_torch():
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    our_probs = softmax(logits)
    torch_probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
    assert np.allclose(our_probs, torch_probs, atol=1e-6)
    print("Softmax matches torch.nn.functional.softmax exactly.")
    print("Probabilities:", our_probs.round(4), "\n")


def demo_full_forward_pass_vs_torch():
    rng = np.random.default_rng(42)

    # Small network: 8x8x1 input -> Conv(4 filters, 3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> FC(3) -> Softmax
    x = rng.normal(size=(8, 8, 1)).astype(np.float32)
    filters = rng.normal(size=(4, 3, 3, 1)).astype(np.float32) * 0.5
    conv_bias = rng.normal(size=4).astype(np.float32) * 0.1

    conv_out_h = 8 - 3 + 1   # 6
    pool_out_h = (conv_out_h - 2) // 2 + 1  # 3
    flat_size = pool_out_h * pool_out_h * 4  # 3*3*4 = 36

    W2 = rng.normal(size=(3, flat_size)).astype(np.float32) * 0.1
    b2 = rng.normal(size=3).astype(np.float32) * 0.1

    y_hat, cache = forward_pass(x, filters, conv_bias, W2, b2)
    print("From-scratch forward pass output (softmax probabilities):", y_hat.round(6))

    # Build an equivalent torch model and copy in the exact same weights
    torch_x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, 1, 8, 8)
    conv = nn.Conv2d(1, 4, kernel_size=3)
    with torch.no_grad():
        # our filters are (K, f, f, C_in); torch wants (out_c, in_c, f, f)
        torch_filters = torch.from_numpy(filters).permute(0, 3, 1, 2)
        conv.weight.copy_(torch_filters)
        conv.bias.copy_(torch.from_numpy(conv_bias))

    z1 = conv(torch_x)
    a1 = torch.relu(z1)
    p1 = F.max_pool2d(a1, kernel_size=2, stride=2)
    # our numpy p1 is (H, W, C) row-major; torch's is (1, C, H, W) -> permute to match flatten order
    f1 = p1.permute(0, 2, 3, 1).flatten()

    fc = nn.Linear(flat_size, 3)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(W2))
        fc.bias.copy_(torch.from_numpy(b2))

    z2 = fc(f1)
    torch_y_hat = F.softmax(z2, dim=0).detach().numpy()

    print("Torch equivalent forward pass output:                  ", torch_y_hat.round(6))
    assert np.allclose(y_hat, torch_y_hat, atol=1e-4)
    print("\nMatch confirmed: from-scratch forward pass agrees with PyTorch.")


def demo_batch_forward_pass():
    rng = np.random.default_rng(0)
    batch_size = 5
    filters = rng.normal(size=(4, 3, 3, 1)).astype(np.float32) * 0.5
    conv_bias = rng.normal(size=4).astype(np.float32) * 0.1
    flat_size = 3 * 3 * 4
    W2 = rng.normal(size=(3, flat_size)).astype(np.float32) * 0.1
    b2 = rng.normal(size=3).astype(np.float32) * 0.1

    outputs = []
    for _ in range(batch_size):
        x = rng.normal(size=(8, 8, 1)).astype(np.float32)
        y_hat, _ = forward_pass(x, filters, conv_bias, W2, b2)
        outputs.append(y_hat)
    outputs = np.stack(outputs)
    print(f"\nBatch forward pass output shape: {outputs.shape} (expected ({batch_size}, 3))")
    assert outputs.shape == (batch_size, 3)


if __name__ == "__main__":
    print("=== Softmax vs torch ===")
    demo_softmax_matches_torch()

    print("=== Full forward pass vs equivalent torch model ===")
    demo_full_forward_pass_vs_torch()

    print("=== Batch forward pass ===")
    demo_batch_forward_pass()
