"""
A single RNN cell implemented from scratch with NumPy, verified against
torch.nn.RNNCell, plus a demo of the hidden state evolving over several
time steps.
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    z = W_xh @ x_t + W_hh @ h_prev + b_h
    return np.tanh(z)


def demo_vs_torch():
    rng = np.random.default_rng(0)
    input_size, hidden_size = 4, 3

    W_xh = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.5
    W_hh = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.5
    b_h = rng.normal(size=hidden_size).astype(np.float32) * 0.1

    x_t = rng.normal(size=input_size).astype(np.float32)
    h_prev = rng.normal(size=hidden_size).astype(np.float32)

    our_h_t = rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h)

    torch_cell = nn.RNNCell(input_size, hidden_size)
    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.from_numpy(W_xh))
        torch_cell.weight_hh.copy_(torch.from_numpy(W_hh))
        torch_cell.bias_ih.zero_()
        torch_cell.bias_hh.copy_(torch.from_numpy(b_h))

    torch_h_t = torch_cell(
        torch.from_numpy(x_t).unsqueeze(0), torch.from_numpy(h_prev).unsqueeze(0)
    ).squeeze(0).detach().numpy()

    print("Our from-scratch h_t:", our_h_t.round(5))
    print("torch.nn.RNNCell h_t:", torch_h_t.round(5))
    assert np.allclose(our_h_t, torch_h_t, atol=1e-5)
    print("Match confirmed.\n")


def demo_hidden_state_evolution():
    rng = np.random.default_rng(1)
    input_size, hidden_size = 2, 3

    W_xh = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.6
    W_hh = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.6
    b_h = np.zeros(hidden_size, dtype=np.float32)

    sequence = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ]

    h = np.zeros(hidden_size, dtype=np.float32)
    print("h_0:", h.round(4))
    for t, x_t in enumerate(sequence, start=1):
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        print(f"h_{t} (after input {sequence[t-1]}):", h.round(4))

    print("\nEach hidden state differs from the last -- the cell is combining")
    print("the new input with the running summary of everything before it.\n")


def demo_zero_weight_ablations():
    rng = np.random.default_rng(2)
    input_size, hidden_size = 2, 3
    W_xh = rng.normal(size=(hidden_size, input_size)).astype(np.float32)
    W_hh = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32)
    b_h = np.zeros(hidden_size, dtype=np.float32)

    x1 = np.array([1.0, 0.0], dtype=np.float32)
    x2 = np.array([0.0, 1.0], dtype=np.float32)

    # Ablation 1: W_hh = 0 -> no memory carried forward, h_t depends only on x_t
    W_hh_zero = np.zeros_like(W_hh)
    h_prev = rng.normal(size=hidden_size).astype(np.float32)  # some arbitrary "memory"
    h_a = rnn_cell_forward(x1, h_prev, W_xh, W_hh_zero, b_h)
    h_b = rnn_cell_forward(x1, np.zeros(hidden_size, dtype=np.float32), W_xh, W_hh_zero, b_h)
    print("W_hh = 0: hidden state ignores h_prev entirely.")
    print("  Same x_t, different h_prev -> identical output?", np.allclose(h_a, h_b))

    # Ablation 2: W_xh = 0 -> hidden state cannot respond to input at all
    W_xh_zero = np.zeros_like(W_xh)
    h_c = rnn_cell_forward(x1, h_prev, W_xh_zero, W_hh, b_h)
    h_d = rnn_cell_forward(x2, h_prev, W_xh_zero, W_hh, b_h)
    print("W_xh = 0: hidden state ignores x_t entirely.")
    print("  Different x_t, same h_prev -> identical output?", np.allclose(h_c, h_d))


if __name__ == "__main__":
    print("=== RNN cell vs torch.nn.RNNCell ===")
    demo_vs_torch()

    print("=== Hidden state evolving over time ===")
    demo_hidden_state_evolution()

    print("=== Zero-weight ablations ===")
    demo_zero_weight_ablations()
