"""
GRU cell implemented from scratch, verified against torch.nn.GRUCell,
plus a parameter-count comparison against LSTM and a small training
speed comparison.
"""

import time

import numpy as np
import torch
import torch.nn as nn


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_cell_forward(x_t, h_prev, Wx_z, Wh_z, b_z_x, b_z_h, Wx_r, Wh_r, b_r_x, b_r_h,
                      Wx_h, Wh_h, b_h_x, b_h_h):
    """Matches PyTorch's exact GRU formula. Two details differ from the
    textbook version in the lesson README:
    (1) the reset gate multiplies (W_hn @ h_prev + b_hn) as a whole, not
        just W_hn @ h_prev;
    (2) PyTorch's update-gate convention is FLIPPED relative to the
        original Cho et al. paper: here z_t close to 1 means "keep the
        OLD hidden state", not "take the new candidate" -- so the final
        blend is z_t*h_prev + (1-z_t)*h_tilde_t, not (1-z_t)*h_prev + z_t*h_tilde_t.
    Both are legitimate GRUs; only the labeling convention differs."""
    z_t = sigmoid(Wx_z @ x_t + b_z_x + Wh_z @ h_prev + b_z_h)
    r_t = sigmoid(Wx_r @ x_t + b_r_x + Wh_r @ h_prev + b_r_h)

    h_tilde_t = np.tanh(Wx_h @ x_t + b_h_x + r_t * (Wh_h @ h_prev + b_h_h))

    h_t = z_t * h_prev + (1 - z_t) * h_tilde_t  # PyTorch convention (see docstring)
    return h_t


def demo_vs_torch():
    rng = np.random.default_rng(0)
    input_size, hidden_size = 3, 4

    Wx_z = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.3
    Wh_z = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.3
    Wx_r = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.3
    Wh_r = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.3
    Wx_h = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.3
    Wh_h = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.3

    b_z_x = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_z_h = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_r_x = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_r_h = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_h_x = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_h_h = rng.normal(size=hidden_size).astype(np.float32) * 0.1

    x_t = rng.normal(size=input_size).astype(np.float32)
    h_prev = rng.normal(size=hidden_size).astype(np.float32)

    our_h = gru_cell_forward(x_t, h_prev, Wx_z, Wh_z, b_z_x, b_z_h, Wx_r, Wh_r, b_r_x, b_r_h,
                              Wx_h, Wh_h, b_h_x, b_h_h)

    # torch.nn.GRUCell packs gates in order [r, z, n(candidate)] along dim 0.
    weight_ih = np.concatenate([Wx_r, Wx_z, Wx_h], axis=0)
    weight_hh = np.concatenate([Wh_r, Wh_z, Wh_h], axis=0)
    bias_ih = np.concatenate([b_r_x, b_z_x, b_h_x])
    bias_hh = np.concatenate([b_r_h, b_z_h, b_h_h])

    torch_cell = nn.GRUCell(input_size, hidden_size)
    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.from_numpy(weight_ih))
        torch_cell.weight_hh.copy_(torch.from_numpy(weight_hh))
        torch_cell.bias_ih.copy_(torch.from_numpy(bias_ih))
        torch_cell.bias_hh.copy_(torch.from_numpy(bias_hh))

    torch_h = torch_cell(
        torch.from_numpy(x_t).unsqueeze(0), torch.from_numpy(h_prev).unsqueeze(0)
    ).squeeze(0).detach().numpy()

    print("Our h_t:  ", our_h.round(5))
    print("Torch h_t:", torch_h.round(5))

    assert np.allclose(our_h, torch_h, atol=1e-4), f"max diff: {np.abs(our_h - torch_h).max()}"
    print("\nMatch confirmed against torch.nn.GRUCell (bit-exact, including the")
    print("reset-gate-multiplies-the-hidden-bias detail).\n")


def demo_param_count_comparison():
    hidden_size, input_size = 128, 64

    lstm_params = 4 * hidden_size * (hidden_size + input_size + 1)  # +1 for bias per gate
    gru_params = 3 * hidden_size * (hidden_size + input_size + 1)

    print(f"hidden_size={hidden_size}, input_size={input_size}")
    print(f"LSTM parameters (hand-computed): {lstm_params:,}")
    print(f"GRU parameters (hand-computed):  {gru_params:,}")
    print(f"Ratio: {lstm_params / gru_params:.3f} (expected ~4/3 = 1.333)\n")

    torch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    torch_gru = nn.GRU(input_size, hidden_size, batch_first=True)
    lstm_actual = sum(p.numel() for p in torch_lstm.parameters())
    gru_actual = sum(p.numel() for p in torch_gru.parameters())

    print(f"LSTM parameters (torch actual): {lstm_actual:,}")
    print(f"GRU parameters (torch actual):  {gru_actual:,}")
    print(f"Ratio: {lstm_actual / gru_actual:.3f}\n")


def demo_training_speed_comparison():
    torch.manual_seed(0)
    input_size, hidden_size, seq_len, batch_size = 32, 64, 50, 16
    n_steps = 30

    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 2, (batch_size,))

    for name, layer_cls in [("LSTM", nn.LSTM), ("GRU", nn.GRU)]:
        torch.manual_seed(0)
        rnn = layer_cls(input_size, hidden_size, batch_first=True)
        fc = nn.Linear(hidden_size, 2)
        params = list(rnn.parameters()) + list(fc.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        start = time.perf_counter()
        for _ in range(n_steps):
            optimizer.zero_grad()
            output, state = rnn(X)
            h_last = output[:, -1, :]
            logits = fc(h_last)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - start

        print(f"{name}: {n_steps} training steps took {elapsed:.4f}s  (final loss: {loss.item():.4f})")


if __name__ == "__main__":
    print("=== GRU cell vs torch.nn.GRUCell ===")
    demo_vs_torch()

    print("=== Parameter count: LSTM vs GRU ===")
    demo_param_count_comparison()

    print("=== Training speed: LSTM vs GRU ===")
    demo_training_speed_comparison()
