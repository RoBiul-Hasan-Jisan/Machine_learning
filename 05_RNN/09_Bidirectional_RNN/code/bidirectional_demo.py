"""
Bidirectional RNN implemented from scratch (two independent RNN cell
passes, concatenated), verified against torch.nn.RNN(bidirectional=True),
plus a demonstration that the backward direction depends on future input.
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)


def bidirectional_rnn_forward(sequence, W_xh_f, W_hh_f, b_h_f, W_xh_b, W_hh_b, b_h_b):
    hidden_size = W_hh_f.shape[0]

    # Forward pass
    h = np.zeros(hidden_size, dtype=np.float32)
    forward_states = []
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh_f, W_hh_f, b_h_f)
        forward_states.append(h.copy())

    # Backward pass (over the reversed sequence)
    h = np.zeros(hidden_size, dtype=np.float32)
    backward_states_reversed = []
    for x_t in reversed(sequence):
        h = rnn_cell_forward(x_t, h, W_xh_b, W_hh_b, b_h_b)
        backward_states_reversed.append(h.copy())
    backward_states = list(reversed(backward_states_reversed))

    combined = [np.concatenate([f, b]) for f, b in zip(forward_states, backward_states)]
    return combined, forward_states, backward_states


def demo_vs_torch():
    rng = np.random.default_rng(0)
    input_size, hidden_size, T = 3, 4, 5

    W_xh_f = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.4
    W_hh_f = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    b_h_f = rng.normal(size=hidden_size).astype(np.float32) * 0.1

    W_xh_b = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.4
    W_hh_b = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    b_h_b = rng.normal(size=hidden_size).astype(np.float32) * 0.1

    sequence = [rng.normal(size=input_size).astype(np.float32) for _ in range(T)]

    combined, our_fwd, our_bwd = bidirectional_rnn_forward(
        sequence, W_xh_f, W_hh_f, b_h_f, W_xh_b, W_hh_b, b_h_b
    )

    torch_rnn = nn.RNN(input_size, hidden_size, bidirectional=True, batch_first=True)
    with torch.no_grad():
        # forward direction: suffix _l0; backward direction: suffix _l0_reverse
        torch_rnn.weight_ih_l0.copy_(torch.from_numpy(W_xh_f))
        torch_rnn.weight_hh_l0.copy_(torch.from_numpy(W_hh_f))
        torch_rnn.bias_ih_l0.zero_()
        torch_rnn.bias_hh_l0.copy_(torch.from_numpy(b_h_f))

        torch_rnn.weight_ih_l0_reverse.copy_(torch.from_numpy(W_xh_b))
        torch_rnn.weight_hh_l0_reverse.copy_(torch.from_numpy(W_hh_b))
        torch_rnn.bias_ih_l0_reverse.zero_()
        torch_rnn.bias_hh_l0_reverse.copy_(torch.from_numpy(b_h_b))

    x_torch = torch.from_numpy(np.stack(sequence)).unsqueeze(0)
    torch_output, _ = torch_rnn(x_torch)  # (1, T, 2*hidden_size)
    torch_output = torch_output.squeeze(0).detach().numpy()

    our_combined = np.stack(combined)
    print("Our combined output shape:  ", our_combined.shape)
    print("Torch combined output shape:", torch_output.shape)

    assert np.allclose(our_combined, torch_output, atol=1e-4), \
        f"max diff: {np.abs(our_combined - torch_output).max()}"
    print("\nMatch confirmed against torch.nn.RNN(bidirectional=True).\n")


def demo_backward_depends_on_future():
    rng = np.random.default_rng(1)
    input_size, hidden_size = 2, 3

    W_xh_f = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.4
    W_hh_f = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    b_h_f = np.zeros(hidden_size, dtype=np.float32)
    W_xh_b = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.4
    W_hh_b = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    b_h_b = np.zeros(hidden_size, dtype=np.float32)

    sequence_a = [rng.normal(size=input_size).astype(np.float32) for _ in range(4)]
    sequence_b = [x.copy() for x in sequence_a]
    sequence_b[-1] = sequence_b[-1] + 5.0  # change ONLY the LAST input

    _, fwd_a, bwd_a = bidirectional_rnn_forward(sequence_a, W_xh_f, W_hh_f, b_h_f, W_xh_b, W_hh_b, b_h_b)
    _, fwd_b, bwd_b = bidirectional_rnn_forward(sequence_b, W_xh_f, W_hh_f, b_h_f, W_xh_b, W_hh_b, b_h_b)

    # Forward direction's hidden state at position 0 should be UNCHANGED
    # (it never saw the last input yet at that point).
    fwd_diff_at_0 = np.abs(fwd_a[0] - fwd_b[0]).max()
    # Backward direction's hidden state at position 0 SHOULD change
    # (the backward pass starts from the end, so it saw the changed last input).
    bwd_diff_at_0 = np.abs(bwd_a[0] - bwd_b[0]).max()

    print(f"Forward h at position 0, difference when changing the LAST input: {fwd_diff_at_0:.8f}")
    print(f"Backward h at position 0, difference when changing the LAST input: {bwd_diff_at_0:.4f}")

    assert fwd_diff_at_0 < 1e-6
    assert bwd_diff_at_0 > 1e-3
    print("\nConfirmed: the forward direction is unaffected by a change to a later")
    print("input (it hasn't 'seen' it yet), while the backward direction IS")
    print("affected -- exactly the future-context access bidirectionality provides.")


if __name__ == "__main__":
    print("=== Bidirectional RNN vs torch.nn.RNN(bidirectional=True) ===")
    demo_vs_torch()

    print("=== Backward direction depends on future input ===")
    demo_backward_depends_on_future()
