"""
Strided, padded convolution from scratch, with output-size verification
against the formula and against torch.nn.Conv2d.
"""

import numpy as np
import torch
import torch.nn as nn


def output_size(n, f, s, p):
    return (n + 2 * p - f) // s + 1


def pad_input(input_matrix, p):
    if p == 0:
        return input_matrix
    return np.pad(input_matrix, pad_width=p, mode="constant", constant_values=0)


def convolve2d_strided(input_matrix, kernel, stride=1, padding=0):
    padded = pad_input(input_matrix, padding)
    ih, iw = padded.shape
    kh, kw = kernel.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1

    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            row, col = i * stride, j * stride
            patch = padded[row:row + kh, col:col + kw]
            output[i, j] = np.sum(patch * kernel)
    return output


def demo_output_size_formula():
    cases = [
        (32, 5, 1, 0),   # valid
        (32, 5, 1, 2),   # same
        (7, 3, 2, 0),
        (28, 3, 2, 1),
    ]
    for n, f, s, p in cases:
        expected = output_size(n, f, s, p)
        rng = np.random.default_rng(0)
        input_matrix = rng.normal(size=(n, n))
        kernel = rng.normal(size=(f, f))
        actual = convolve2d_strided(input_matrix, kernel, stride=s, padding=p).shape[0]
        status = "OK" if actual == expected else "MISMATCH"
        print(f"n={n:3d} f={f} s={s} p={p} -> formula={expected:3d}, actual={actual:3d}  [{status}]")


def demo_same_padding_stack():
    """Stack 5 conv layers with 'same' padding (stride 1); spatial size unchanged."""
    x = torch.randn(1, 3, 32, 32)
    layers = nn.Sequential(*[
        nn.Conv2d(3 if i == 0 else 16, 16, kernel_size=3, stride=1, padding=1)
        for i in range(5)
    ])
    out = layers(x)
    print(f"\nInput spatial size: {x.shape[2]}x{x.shape[3]}")
    print(f"Output spatial size after 5 'same'-padded conv layers: {out.shape[2]}x{out.shape[3]}")
    assert out.shape[2] == x.shape[2] and out.shape[3] == x.shape[3]
    print("Confirmed: spatial size preserved by 'same' padding.\n")


def demo_stride2_downsampling():
    x = torch.randn(1, 3, 32, 32)
    conv_stride2 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    out = conv_stride2(x)
    print(f"Input spatial size: {x.shape[2]}x{x.shape[3]}")
    print(f"Output spatial size after ONE stride-2 conv layer: {out.shape[2]}x{out.shape[3]}")
    print("Roughly halved, as expected for stride 2.")


if __name__ == "__main__":
    print("=== Output size formula verification ===")
    demo_output_size_formula()

    print("\n=== 'Same' padding keeps spatial size constant across layers ===")
    demo_same_padding_stack()

    print("=== Stride 2 halves spatial size in one layer ===")
    demo_stride2_downsampling()
