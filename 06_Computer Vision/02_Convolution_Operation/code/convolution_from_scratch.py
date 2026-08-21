"""
2D convolution from scratch: naive triple-loop version, a vectorized
im2col version, and a numerical comparison against torch's conv2d.
"""

import numpy as np
import torch
import torch.nn.functional as F


def convolve2d(input_matrix, kernel):
    """Single-channel 2D convolution (cross-correlation), stride 1, no padding."""
    ih, iw = input_matrix.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1

    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = input_matrix[i:i + kh, j:j + kw]
            output[i, j] = np.sum(patch * kernel)
    return output


def convolve2d_multichannel(input_tensor, kernel):
    """input_tensor: (H, W, C_in). kernel: (f, f, C_in). Returns (H_out, W_out)."""
    ih, iw, c = input_tensor.shape
    kh, kw, kc = kernel.shape
    assert c == kc, "channel mismatch"
    oh, ow = ih - kh + 1, iw - kw + 1

    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = input_tensor[i:i + kh, j:j + kw, :]
            output[i, j] = np.sum(patch * kernel)
    return output


def conv_layer(input_tensor, filters):
    """input_tensor: (H, W, C_in). filters: (K, f, f, C_in). Returns (H_out, W_out, K)."""
    k = filters.shape[0]
    outputs = [convolve2d_multichannel(input_tensor, filters[i]) for i in range(k)]
    return np.stack(outputs, axis=-1)


def im2col_convolve2d(input_matrix, kernel):
    """Vectorized single-channel convolution using the im2col trick."""
    ih, iw = input_matrix.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1

    # Build a matrix where each row is one flattened receptive-field patch
    cols = np.zeros((oh * ow, kh * kw))
    row = 0
    for i in range(oh):
        for j in range(ow):
            patch = input_matrix[i:i + kh, j:j + kw]
            cols[row, :] = patch.flatten()
            row += 1

    kernel_flat = kernel.flatten()
    output_flat = cols @ kernel_flat        # single matrix-vector multiply
    return output_flat.reshape(oh, ow)


def demo_hand_example():
    input_matrix = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 0],
        [1, 0, 1, 2, 0],
        [2, 3, 0, 1, 2],
        [0, 1, 2, 0, 1],
    ], dtype=float)

    kernel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ], dtype=float)

    output = convolve2d(input_matrix, kernel)
    print("Input:\n", input_matrix)
    print("Kernel:\n", kernel)
    print("Output feature map:\n", output)
    assert output[0, 0] == 7.0, "hand-computed example should give 7 at (0,0)"
    print("Matches the hand-computed value at (0,0) = 7.\n")


def demo_naive_vs_im2col():
    rng = np.random.default_rng(0)
    input_matrix = rng.normal(size=(20, 20))
    kernel = rng.normal(size=(3, 3))

    out_naive = convolve2d(input_matrix, kernel)
    out_im2col = im2col_convolve2d(input_matrix, kernel)
    assert np.allclose(out_naive, out_im2col), "naive and im2col results should match"
    print("Naive and im2col convolution results match exactly.\n")


def demo_multichannel_layer():
    rng = np.random.default_rng(1)
    input_tensor = rng.normal(size=(10, 10, 3))       # H, W, C_in
    filters = rng.normal(size=(4, 3, 3, 3))           # K, f, f, C_in

    output = conv_layer(input_tensor, filters)
    print("Multi-channel conv layer output shape:", output.shape)
    assert output.shape == (8, 8, 4)
    print("Matches expected (H_out, W_out, K) = (8, 8, 4)\n")


def demo_vs_torch():
    rng = np.random.default_rng(2)
    input_matrix = rng.normal(size=(10, 10)).astype(np.float32)
    kernel = rng.normal(size=(3, 3)).astype(np.float32)

    our_output = convolve2d(input_matrix, kernel)

    # torch conv2d expects (batch, channels, H, W) and does cross-correlation
    # (same convention as this lesson), with no flipping.
    torch_input = torch.from_numpy(input_matrix).unsqueeze(0).unsqueeze(0)
    torch_kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
    torch_output = F.conv2d(torch_input, torch_kernel).squeeze().numpy()

    assert np.allclose(our_output, torch_output, atol=1e-4)
    print("Our from-scratch convolution matches torch.nn.functional.conv2d exactly.")


if __name__ == "__main__":
    print("=== Hand-computed example ===")
    demo_hand_example()

    print("=== Naive vs im2col (should be identical) ===")
    demo_naive_vs_im2col()

    print("=== Multi-channel, multi-filter conv layer ===")
    demo_multichannel_layer()

    print("=== Comparison against torch.nn.functional.conv2d ===")
    demo_vs_torch()
