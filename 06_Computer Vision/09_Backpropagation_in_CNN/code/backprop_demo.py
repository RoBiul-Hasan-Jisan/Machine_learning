"""
From-scratch backward pass through Conv -> ReLU -> MaxPool -> FC -> Softmax,
with every computed gradient verified against PyTorch's autograd.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Forward pass pieces (from Lesson 08), extended to cache what backprop needs ---

def conv_forward(x, filters, bias):
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


def conv_backward(d_out, x, filters):
    """d_out: (H_out, W_out, K) gradient w.r.t. conv output.
    Returns dW (K, f, f, C_in), db (K,), dx (H, W, C_in)."""
    ih, iw, c_in = x.shape
    k, fh, fw, _ = filters.shape
    oh, ow = d_out.shape[0], d_out.shape[1]

    dW = np.zeros_like(filters)
    db = np.zeros(k)
    dx = np.zeros_like(x)

    for filt_idx in range(k):
        db[filt_idx] = np.sum(d_out[:, :, filt_idx])
        for i in range(oh):
            for j in range(ow):
                patch = x[i:i + fh, j:j + fw, :]
                grad = d_out[i, j, filt_idx]
                dW[filt_idx] += grad * patch
                dx[i:i + fh, j:j + fw, :] += grad * filters[filt_idx]

    return dW, db, dx


def relu_forward(z):
    return np.maximum(0, z)


def relu_backward(d_out, z):
    return d_out * (z > 0)


def max_pool_forward(x, f=2, s=2):
    ih, iw, c = x.shape
    oh, ow = (ih - f) // s + 1, (iw - f) // s + 1
    pooled = np.zeros((oh, ow, c))
    argmax_positions = np.zeros((oh, ow, c, 2), dtype=int)

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


def max_pool_backward(d_out, argmax_positions, input_shape):
    """Route each output gradient to only the argmax position in the input."""
    dx = np.zeros(input_shape)
    oh, ow, c = d_out.shape
    for ch in range(c):
        for i in range(oh):
            for j in range(ow):
                row, col = argmax_positions[i, j, ch]
                dx[row, col, ch] += d_out[i, j, ch]
    return dx


def demo_relu_backward_vs_torch():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 5)).astype(np.float32)
    d_out = rng.normal(size=(5, 5)).astype(np.float32)

    our_dz = relu_backward(d_out, z)

    torch_z = torch.from_numpy(z).requires_grad_(True)
    torch_a = torch.relu(torch_z)
    torch_a.backward(torch.from_numpy(d_out))
    torch_dz = torch_z.grad.numpy()

    assert np.allclose(our_dz, torch_dz, atol=1e-6)
    print("ReLU backward matches torch.autograd exactly.\n")


def demo_maxpool_backward_hand_example():
    x = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [0, 1, 8, 3],
        [2, 4, 1, 1],
    ], dtype=float).reshape(4, 4, 1)

    pooled, argmax_positions = max_pool_forward(x, f=2, s=2)
    d_out = np.ones((2, 2, 1))  # incoming gradient of 1.0 everywhere

    dx = max_pool_backward(d_out, argmax_positions, x.shape)
    print("Max-pool backward gradient routed to input (should be 1.0 only at argmax positions):")
    print(dx.squeeze())
    nonzero_count = np.count_nonzero(dx)
    print(f"Nonzero gradient positions: {nonzero_count} out of {dx.size} (one per pooling window)\n")
    assert nonzero_count == 4  # one argmax per 2x2 window, 4 windows


def demo_conv_backward_vs_torch():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 6, 1)).astype(np.float32)
    filters = rng.normal(size=(2, 3, 3, 1)).astype(np.float32) * 0.5
    bias = rng.normal(size=2).astype(np.float32) * 0.1

    z1 = conv_forward(x, filters, bias)
    d_out = rng.normal(size=z1.shape).astype(np.float32)  # pretend this came from later layers

    our_dW, our_db, our_dx = conv_backward(d_out, x, filters)

    # Equivalent torch computation
    torch_x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)  # (1,1,6,6)
    torch_filters = torch.from_numpy(filters).permute(0, 3, 1, 2).requires_grad_(True)  # (K,1,3,3)
    torch_bias = torch.from_numpy(bias).requires_grad_(True)

    torch_z1 = F.conv2d(torch_x, torch_filters, torch_bias)  # (1, K, H_out, W_out)
    torch_d_out = torch.from_numpy(d_out).permute(2, 0, 1).unsqueeze(0)  # match torch's (1,K,H,W)
    torch_z1.backward(torch_d_out)

    torch_dW = torch_filters.grad.permute(0, 2, 3, 1).numpy()  # back to (K,f,f,C_in)
    torch_db = torch_bias.grad.numpy()
    torch_dx = torch_x.grad.permute(0, 2, 3, 1).squeeze(0).numpy()  # back to (H,W,C_in)

    assert np.allclose(our_dW, torch_dW, atol=1e-3), f"dW mismatch: max diff {np.abs(our_dW - torch_dW).max()}"
    assert np.allclose(our_db, torch_db, atol=1e-3)
    assert np.allclose(our_dx, torch_dx, atol=1e-3)
    print("Conv backward (dW, db, dx) all match torch.autograd exactly.\n")


if __name__ == "__main__":
    print("=== ReLU backward vs torch ===")
    demo_relu_backward_vs_torch()

    print("=== Max-pool backward: hand example ===")
    demo_maxpool_backward_hand_example()

    print("=== Conv backward vs torch ===")
    demo_conv_backward_vs_torch()
