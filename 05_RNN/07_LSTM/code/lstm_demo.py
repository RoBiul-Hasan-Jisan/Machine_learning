"""
LSTM cell implemented from scratch, verified against torch.nn.LSTMCell,
plus a gradient-magnitude comparison between a plain RNN and an LSTM
over a long sequence.
"""

import numpy as np
import torch
import torch.nn as nn


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_cell_forward(x_t, h_prev, c_prev, W_f, b_f, W_i, b_i, W_o, b_o, W_c, b_c):
    combined = np.concatenate([h_prev, x_t])

    f_t = sigmoid(W_f @ combined + b_f)
    i_t = sigmoid(W_i @ combined + b_i)
    o_t = sigmoid(W_o @ combined + b_o)
    c_tilde_t = np.tanh(W_c @ combined + b_c)

    c_t = f_t * c_prev + i_t * c_tilde_t
    h_t = o_t * np.tanh(c_t)

    return h_t, c_t


def demo_vs_torch():
    rng = np.random.default_rng(0)
    input_size, hidden_size = 3, 4
    combined_size = input_size + hidden_size

    W_f = rng.normal(size=(hidden_size, combined_size)).astype(np.float32) * 0.3
    W_i = rng.normal(size=(hidden_size, combined_size)).astype(np.float32) * 0.3
    W_o = rng.normal(size=(hidden_size, combined_size)).astype(np.float32) * 0.3
    W_c = rng.normal(size=(hidden_size, combined_size)).astype(np.float32) * 0.3
    b_f = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_i = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_o = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    b_c = rng.normal(size=hidden_size).astype(np.float32) * 0.1

    x_t = rng.normal(size=input_size).astype(np.float32)
    h_prev = rng.normal(size=hidden_size).astype(np.float32)
    c_prev = rng.normal(size=hidden_size).astype(np.float32)

    our_h, our_c = lstm_cell_forward(x_t, h_prev, c_prev, W_f, b_f, W_i, b_i, W_o, b_o, W_c, b_c)

    # torch.nn.LSTMCell packs gates in order [i, f, g, o] along dim 0,
    # each split of size hidden_size, and splits weight_ih (from x) / weight_hh (from h).
    torch_cell = nn.LSTMCell(input_size, hidden_size)
    # Our combined = [h_prev, x_t], so first `hidden_size` cols act on h, rest on x.
    Wh_f, Wx_f = W_f[:, :hidden_size], W_f[:, hidden_size:]
    Wh_i, Wx_i = W_i[:, :hidden_size], W_i[:, hidden_size:]
    Wh_o, Wx_o = W_o[:, :hidden_size], W_o[:, hidden_size:]
    Wh_c, Wx_c = W_c[:, :hidden_size], W_c[:, hidden_size:]

    weight_ih = np.concatenate([Wx_i, Wx_f, Wx_c, Wx_o], axis=0)  # order: i, f, g, o
    weight_hh = np.concatenate([Wh_i, Wh_f, Wh_c, Wh_o], axis=0)
    bias_ih = np.concatenate([b_i, b_f, b_c, b_o])
    bias_hh = np.zeros_like(bias_ih)  # put all bias into bias_ih to avoid double-counting

    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.from_numpy(weight_ih))
        torch_cell.weight_hh.copy_(torch.from_numpy(weight_hh))
        torch_cell.bias_ih.copy_(torch.from_numpy(bias_ih))
        torch_cell.bias_hh.copy_(torch.from_numpy(bias_hh))

    torch_h, torch_c = torch_cell(
        torch.from_numpy(x_t).unsqueeze(0),
        (torch.from_numpy(h_prev).unsqueeze(0), torch.from_numpy(c_prev).unsqueeze(0)),
    )
    torch_h = torch_h.squeeze(0).detach().numpy()
    torch_c = torch_c.squeeze(0).detach().numpy()

    print("Our h_t:  ", our_h.round(5))
    print("Torch h_t:", torch_h.round(5))
    print("Our c_t:  ", our_c.round(5))
    print("Torch c_t:", torch_c.round(5))

    assert np.allclose(our_h, torch_h, atol=1e-4)
    assert np.allclose(our_c, torch_c, atol=1e-4)
    print("\nMatch confirmed against torch.nn.LSTMCell.\n")


def demo_forget_gate_bias_trick():
    print("sigmoid(0)  =", round(sigmoid(np.array(0.0)).item(), 4), " -> roughly half the cell state kept")
    print("sigmoid(5)  =", round(sigmoid(np.array(5.0)).item(), 4), " -> almost all of the cell state kept")
    print("\nInitializing the forget gate's bias to a large positive value (e.g. 5)")
    print("means, at the very start of training, f_t starts close to 1 for every")
    print("step -- the cell state is preserved by default rather than discarded,")
    print("giving gradients a clear path backward from the first training step,")
    print("rather than the network needing to first LEARN to stop forgetting.\n")


def gradient_norm_at_input(seq_len, cell_type, hidden_size=8, input_size=4, seed=0):
    torch.manual_seed(seed)
    if cell_type == "rnn":
        layer = nn.RNN(input_size, hidden_size, batch_first=True)
    else:
        layer = nn.LSTM(input_size, hidden_size, batch_first=True)

    x = torch.randn(1, seq_len, input_size, requires_grad=True)
    if cell_type == "rnn":
        _, h_n = layer(x)
        loss = h_n.sum()
    else:
        _, (h_n, c_n) = layer(x)
        loss = h_n.sum() + c_n.sum()
    loss.backward()

    return x.grad[0, 0].norm().item()


def demo_gradient_comparison():
    print("Gradient reaching the FIRST time step's input, plain RNN vs LSTM:")
    print(f"{'seq_len':>8} | {'RNN':>12} | {'LSTM':>12}")
    for seq_len in [10, 30, 60, 100]:
        rnn_grad = gradient_norm_at_input(seq_len, "rnn")
        lstm_grad = gradient_norm_at_input(seq_len, "lstm")
        print(f"{seq_len:8d} | {rnn_grad:12.8f} | {lstm_grad:12.8f}")
    print("\n(LSTM's gradient typically stays substantially larger at long sequence")
    print("lengths, reflecting the more direct gradient path through the cell state.)")


if __name__ == "__main__":
    print("=== LSTM cell vs torch.nn.LSTMCell ===")
    demo_vs_torch()

    print("=== Forget gate bias initialization trick ===")
    demo_forget_gate_bias_trick()

    print("=== Gradient comparison: plain RNN vs LSTM ===")
    demo_gradient_comparison()
