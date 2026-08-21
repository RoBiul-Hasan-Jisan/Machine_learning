"""
A from-scratch 2-layer stacked RNN (layer 2 consumes layer 1's full
output sequence), verified against torch.nn.RNN(num_layers=2), plus a
demonstration of dropout's placement between layers.
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)


def rnn_layer_forward(sequence, W_xh, W_hh, b_h):
    hidden_size = W_hh.shape[0]
    h = np.zeros(hidden_size, dtype=np.float32)
    outputs = []
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        outputs.append(h.copy())
    return outputs, h


def stacked_rnn_forward(sequence, layer_weights):
    """layer_weights: list of (W_xh, W_hh, b_h) tuples, one per layer."""
    current_sequence = sequence
    final_hidden_states = []
    for W_xh, W_hh, b_h in layer_weights:
        current_sequence, h_final = rnn_layer_forward(current_sequence, W_xh, W_hh, b_h)
        final_hidden_states.append(h_final)
    return current_sequence, final_hidden_states  # top layer's outputs, every layer's final h


def demo_vs_torch():
    rng = np.random.default_rng(0)
    input_size, hidden_size, T = 3, 4, 5

    # Layer 1: input_size -> hidden_size
    W_xh_1 = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.4
    W_hh_1 = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    b_h_1 = rng.normal(size=hidden_size).astype(np.float32) * 0.1

    # Layer 2: hidden_size -> hidden_size (its input is layer 1's output)
    W_xh_2 = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    W_hh_2 = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    b_h_2 = rng.normal(size=hidden_size).astype(np.float32) * 0.1

    sequence = [rng.normal(size=input_size).astype(np.float32) for _ in range(T)]

    layer_weights = [(W_xh_1, W_hh_1, b_h_1), (W_xh_2, W_hh_2, b_h_2)]
    our_top_outputs, our_final_states = stacked_rnn_forward(sequence, layer_weights)

    torch_rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
    with torch.no_grad():
        torch_rnn.weight_ih_l0.copy_(torch.from_numpy(W_xh_1))
        torch_rnn.weight_hh_l0.copy_(torch.from_numpy(W_hh_1))
        torch_rnn.bias_ih_l0.zero_()
        torch_rnn.bias_hh_l0.copy_(torch.from_numpy(b_h_1))

        torch_rnn.weight_ih_l1.copy_(torch.from_numpy(W_xh_2))
        torch_rnn.weight_hh_l1.copy_(torch.from_numpy(W_hh_2))
        torch_rnn.bias_ih_l1.zero_()
        torch_rnn.bias_hh_l1.copy_(torch.from_numpy(b_h_2))

    x_torch = torch.from_numpy(np.stack(sequence)).unsqueeze(0)
    torch_output, torch_h_n = torch_rnn(x_torch)
    torch_output = torch_output.squeeze(0).detach().numpy()   # top layer's output sequence
    torch_h_n = torch_h_n.squeeze(1).detach().numpy()          # (num_layers, hidden_size)

    our_top_outputs = np.stack(our_top_outputs)
    our_final_states = np.stack(our_final_states)

    print("Our top-layer output shape:  ", our_top_outputs.shape)
    print("Torch top-layer output shape:", torch_output.shape)
    assert np.allclose(our_top_outputs, torch_output, atol=1e-4)
    print("Top layer's output sequence matches.\n")

    print("Our final hidden states (per layer) shape:  ", our_final_states.shape)
    print("Torch final hidden states (per layer) shape:", torch_h_n.shape)
    assert np.allclose(our_final_states, torch_h_n, atol=1e-4)
    print("Per-layer final hidden states match.\n")

    print("Match confirmed against torch.nn.RNN(num_layers=2).\n")


def demo_dropout_between_layers():
    torch.manual_seed(0)
    input_size, hidden_size, T = 8, 16, 10

    stacked_no_dropout = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.0)
    stacked_with_dropout = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.5)

    x = torch.randn(1, T, input_size)

    stacked_with_dropout.train()
    out_train_1, _ = stacked_with_dropout(x)
    out_train_2, _ = stacked_with_dropout(x)

    stacked_with_dropout.eval()
    out_eval_1, _ = stacked_with_dropout(x)
    out_eval_2, _ = stacked_with_dropout(x)

    print("With inter-layer dropout, train mode: repeated calls differ (stochastic)?",
          not torch.allclose(out_train_1, out_train_2))
    print("With inter-layer dropout, eval mode: repeated calls identical (deterministic)?",
          torch.allclose(out_eval_1, out_eval_2))

    # A single-layer RNN's `dropout` argument has no effect (nothing "between layers" to apply to)
    single_layer = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.5)
    single_layer.train()
    out_single_1, _ = single_layer(x)
    out_single_2, _ = single_layer(x)
    print("\nSingle-layer LSTM with dropout=0.5, train mode: repeated calls identical",
          "(dropout has nowhere to apply)?", torch.allclose(out_single_1, out_single_2))


if __name__ == "__main__":
    print("=== Stacked RNN vs torch.nn.RNN(num_layers=2) ===")
    demo_vs_torch()

    print("=== Dropout between layers ===")
    demo_dropout_between_layers()
