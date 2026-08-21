"""
Max pooling and average pooling from scratch, verified against
torch.nn.MaxPool2d / torch.nn.AvgPool2d, plus a translation-robustness demo.
"""

import numpy as np
import torch
import torch.nn as nn


def max_pool2d(input_matrix, f=2, s=2):
    ih, iw = input_matrix.shape
    oh, ow = (ih - f) // s + 1, (iw - f) // s + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            row, col = i * s, j * s
            window = input_matrix[row:row + f, col:col + f]
            output[i, j] = np.max(window)
    return output


def avg_pool2d(input_matrix, f=2, s=2):
    ih, iw = input_matrix.shape
    oh, ow = (ih - f) // s + 1, (iw - f) // s + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            row, col = i * s, j * s
            window = input_matrix[row:row + f, col:col + f]
            output[i, j] = np.mean(window)
    return output


def demo_hand_example():
    x = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [0, 1, 8, 3],
        [2, 4, 1, 1],
    ], dtype=float)

    max_out = max_pool2d(x, f=2, s=2)
    avg_out = avg_pool2d(x, f=2, s=2)

    print("Input:\n", x)
    print("Max pool (2x2, stride 2):\n", max_out)
    print("Avg pool (2x2, stride 2):\n", avg_out)

    assert np.array_equal(max_out, np.array([[6, 4], [4, 8]]))
    assert np.allclose(avg_out, np.array([[3.75, 2.25], [1.75, 3.25]]))
    print("Matches hand-computed values.\n")


def demo_vs_torch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 10)).astype(np.float32)

    our_max = max_pool2d(x, f=2, s=2)
    our_avg = avg_pool2d(x, f=2, s=2)

    t_x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    torch_max = nn.MaxPool2d(2, stride=2)(t_x).squeeze().numpy()
    torch_avg = nn.AvgPool2d(2, stride=2)(t_x).squeeze().numpy()

    assert np.allclose(our_max, torch_max, atol=1e-5)
    assert np.allclose(our_avg, torch_avg, atol=1e-5)
    print("Our from-scratch pooling matches torch.nn.MaxPool2d / AvgPool2d exactly.\n")


def demo_translation_robustness():
    x = np.zeros((8, 8), dtype=float)
    x[3, 3] = 10.0  # one strong "active" pixel

    x_shifted = np.zeros((8, 8), dtype=float)
    x_shifted[3, 4] = 10.0  # same signal, shifted by 1 pixel

    max_orig = max_pool2d(x, f=2, s=2)
    max_shifted = max_pool2d(x_shifted, f=2, s=2)

    print("Max pool of original active pixel:\n", max_orig)
    print("Max pool after shifting the active pixel by 1 pixel:\n", max_shifted)
    print("Note: within the same 2x2 window, the max is unchanged (partial")
    print("translation robustness); shifting far enough to cross a window")
    print("boundary DOES change the output.")


if __name__ == "__main__":
    print("=== Hand-computed example ===")
    demo_hand_example()

    print("=== Comparison against torch ===")
    demo_vs_torch()

    print("=== Translation robustness demo ===")
    demo_translation_robustness()
