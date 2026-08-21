"""
Full sequence forward propagation: many-to-one and many-to-many
patterns implemented from scratch, verified against torch.nn.RNN,
plus a batched version.
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)


def rnn_forward_many_to_one(sequence, W_xh, W_hh, b_h, W_hy, b_y):
    h = np.zeros(W_hh.shape[0], dtype=np.float32)
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
    y = W_hy @ h + b_y
    return y, h


def rnn_forward_many_to_many(sequence, W_xh, W_hh, b_h, W_hy, b_y):
    h = np.zeros(W_hh.shape[0], dtype=np.float32)
    hidden_states, outputs = [], []
    for x_t in sequence:
        h = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        y_t = W_hy @ h + b_y
        hidden_states.append(h.copy())
        outputs.append(y_t)
    return outputs, hidden_states


def make_weights(input_size, hidden_size, output_size, seed=0):
    rng = np.random.default_rng(seed)
    W_xh = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.5
    W_hh = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.5
    b_h = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    W_hy = rng.normal(size=(output_size, hidden_size)).astype(np.float32) * 0.5
    b_y = rng.normal(size=output_size).astype(np.float32) * 0.1
    return W_xh, W_hh, b_h, W_hy, b_y


def copy_weights_into_torch(torch_rnn, W_xh, W_hh, b_h):
    with torch.no_grad():
        torch_rnn.weight_ih_l0.copy_(torch.from_numpy(W_xh))
        torch_rnn.weight_hh_l0.copy_(torch.from_numpy(W_hh))
        torch_rnn.bias_ih_l0.zero_()
        torch_rnn.bias_hh_l0.copy_(torch.from_numpy(b_h))


def demo_many_to_one_vs_torch():
    input_size, hidden_size, output_size, T = 3, 4, 2, 5
    W_xh, W_hh, b_h, W_hy, b_y = make_weights(input_size, hidden_size, output_size)

    rng = np.random.default_rng(1)
    sequence = [rng.normal(size=input_size).astype(np.float32) for _ in range(T)]

    our_y, our_h_final = rnn_forward_many_to_one(sequence, W_xh, W_hh, b_h, W_hy, b_y)

    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    copy_weights_into_torch(torch_rnn, W_xh, W_hh, b_h)
    fc = nn.Linear(hidden_size, output_size)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(W_hy))
        fc.bias.copy_(torch.from_numpy(b_y))

    x_torch = torch.from_numpy(np.stack(sequence)).unsqueeze(0)  # (1, T, input_size)
    _, h_n = torch_rnn(x_torch)
    torch_y = fc(h_n.squeeze(0).squeeze(0)).detach().numpy()
    torch_h_final = h_n.squeeze(0).squeeze(0).detach().numpy()

    print("Our many-to-one final hidden state:", our_h_final.round(5))
    print("Torch final hidden state:          ", torch_h_final.round(5))
    assert np.allclose(our_h_final, torch_h_final, atol=1e-4)

    print("\nOur many-to-one output:", our_y.round(5))
    print("Torch output:          ", torch_y.round(5))
    assert np.allclose(our_y, torch_y, atol=1e-4)
    print("Match confirmed.\n")


def demo_many_to_many_vs_torch():
    input_size, hidden_size, output_size, T = 3, 4, 2, 5
    W_xh, W_hh, b_h, W_hy, b_y = make_weights(input_size, hidden_size, output_size, seed=2)

    rng = np.random.default_rng(3)
    sequence = [rng.normal(size=input_size).astype(np.float32) for _ in range(T)]

    our_outputs, our_hidden_states = rnn_forward_many_to_many(sequence, W_xh, W_hh, b_h, W_hy, b_y)

    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    copy_weights_into_torch(torch_rnn, W_xh, W_hh, b_h)
    fc = nn.Linear(hidden_size, output_size)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(W_hy))
        fc.bias.copy_(torch.from_numpy(b_y))

    x_torch = torch.from_numpy(np.stack(sequence)).unsqueeze(0)
    all_h, _ = torch_rnn(x_torch)          # (1, T, hidden_size) -- every step's hidden state
    torch_outputs = fc(all_h.squeeze(0)).detach().numpy()  # (T, output_size)

    our_hidden_states = np.stack(our_hidden_states)
    torch_hidden_states = all_h.squeeze(0).detach().numpy()
    assert np.allclose(our_hidden_states, torch_hidden_states, atol=1e-4)
    print("Every step's hidden state matches torch.nn.RNN's full output sequence.")

    our_outputs = np.stack(our_outputs)
    assert np.allclose(our_outputs, torch_outputs, atol=1e-4)
    print("Every step's output matches too. Match confirmed.\n")


def demo_batched_forward():
    """Batch 3 equal-length sequences and confirm results match running
    them one at a time (batching doesn't change per-sequence results)."""
    input_size, hidden_size = 3, 4
    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    rng = np.random.default_rng(4)
    batch = torch.from_numpy(rng.normal(size=(3, 5, input_size)).astype(np.float32))

    batched_output, batched_h_n = torch_rnn(batch)

    for i in range(3):
        single_output, single_h_n = torch_rnn(batch[i:i + 1])
        assert torch.allclose(single_output, batched_output[i:i + 1], atol=1e-5)

    print("Batched forward pass matches running each sequence individually.")
    print(f"Batched output shape: {tuple(batched_output.shape)} (batch, T, hidden_size)")


if __name__ == "__main__":
    print("=== Many-to-one vs torch.nn.RNN ===")
    demo_many_to_one_vs_torch()

    print("=== Many-to-many vs torch.nn.RNN ===")
    demo_many_to_many_vs_torch()

    print("=== Batched forward pass ===")
    demo_batched_forward()
