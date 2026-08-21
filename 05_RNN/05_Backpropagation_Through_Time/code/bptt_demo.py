"""
Backpropagation Through Time (BPTT) implemented from scratch over a
multi-step sequence, with every accumulated gradient verified against
PyTorch's autograd on an equivalent computation. Also demonstrates
truncated BPTT.
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b_h):
    z = W_xh @ x_t + W_hh @ h_prev + b_h
    return np.tanh(z), z


def rnn_backward_step(dh_t, h_t, h_prev, x_t, W_hh):
    dz = dh_t * (1 - h_t ** 2)  # tanh'(z) = 1 - tanh(z)^2 = 1 - h_t^2
    dW_xh = np.outer(dz, x_t)
    dW_hh = np.outer(dz, h_prev)
    db_h = dz
    dh_prev = W_hh.T @ dz
    return dW_xh, dW_hh, db_h, dh_prev


def bptt(sequence, W_xh, W_hh, b_h, dLoss_dh_T):
    """Full (untruncated) BPTT over a sequence, given the gradient of the
    loss w.r.t. the FINAL hidden state (many-to-one style, matching Lesson 04)."""
    hidden_size = W_hh.shape[0]

    # Forward pass, caching every h_t (and h_0)
    h = np.zeros(hidden_size, dtype=np.float32)
    hidden_states = [h.copy()]
    for x_t in sequence:
        h, _ = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        hidden_states.append(h.copy())

    # Backward pass
    dW_xh_total = np.zeros_like(W_xh)
    dW_hh_total = np.zeros_like(W_hh)
    db_h_total = np.zeros_like(b_h)
    dh = dLoss_dh_T.copy()

    for t in reversed(range(len(sequence))):
        h_t = hidden_states[t + 1]
        h_prev = hidden_states[t]
        x_t = sequence[t]
        dW_xh, dW_hh, db_h, dh = rnn_backward_step(dh, h_t, h_prev, x_t, W_hh)
        dW_xh_total += dW_xh
        dW_hh_total += dW_hh
        db_h_total += db_h

    return dW_xh_total, dW_hh_total, db_h_total


def demo_bptt_vs_autograd():
    rng = np.random.default_rng(0)
    input_size, hidden_size, T = 3, 4, 5

    W_xh = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.5
    W_hh = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.5
    b_h = rng.normal(size=hidden_size).astype(np.float32) * 0.1
    sequence = [rng.normal(size=input_size).astype(np.float32) for _ in range(T)]

    dLoss_dh_T = rng.normal(size=hidden_size).astype(np.float32)  # pretend gradient from a loss

    our_dW_xh, our_dW_hh, our_db_h = bptt(sequence, W_xh, W_hh, b_h, dLoss_dh_T)

    # Equivalent computation in torch, using autograd
    torch_W_xh = torch.from_numpy(W_xh).requires_grad_(True)
    torch_W_hh = torch.from_numpy(W_hh).requires_grad_(True)
    torch_b_h = torch.from_numpy(b_h).requires_grad_(True)

    h = torch.zeros(hidden_size)
    for x_t in sequence:
        z = torch_W_xh @ torch.from_numpy(x_t) + torch_W_hh @ h + torch_b_h
        h = torch.tanh(z)

    h.backward(torch.from_numpy(dLoss_dh_T))

    print("dW_xh - ours vs torch, max abs diff:", np.abs(our_dW_xh - torch_W_xh.grad.numpy()).max())
    print("dW_hh - ours vs torch, max abs diff:", np.abs(our_dW_hh - torch_W_hh.grad.numpy()).max())
    print("db_h  - ours vs torch, max abs diff:", np.abs(our_db_h - torch_b_h.grad.numpy()).max())

    assert np.allclose(our_dW_xh, torch_W_xh.grad.numpy(), atol=1e-4)
    assert np.allclose(our_dW_hh, torch_W_hh.grad.numpy(), atol=1e-4)
    assert np.allclose(our_db_h, torch_b_h.grad.numpy(), atol=1e-4)
    print("\nAll gradients match PyTorch autograd exactly.\n")


def demo_per_step_contributions():
    """Isolate each time step's individual contribution to dW_hh, before summing."""
    rng = np.random.default_rng(1)
    input_size, hidden_size, T = 2, 3, 4

    W_xh = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.5
    W_hh = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.5
    b_h = np.zeros(hidden_size, dtype=np.float32)
    sequence = [rng.normal(size=input_size).astype(np.float32) for _ in range(T)]

    h = np.zeros(hidden_size, dtype=np.float32)
    hidden_states = [h.copy()]
    for x_t in sequence:
        h, _ = rnn_cell_forward(x_t, h, W_xh, W_hh, b_h)
        hidden_states.append(h.copy())

    dh = np.ones(hidden_size, dtype=np.float32)  # arbitrary gradient from the loss
    print("Per-step contribution to dW_hh (Frobenius norm of each step's piece):")
    for t in reversed(range(T)):
        h_t = hidden_states[t + 1]
        h_prev = hidden_states[t]
        x_t = sequence[t]
        _, dW_hh_step, _, dh = rnn_backward_step(dh, h_t, h_prev, x_t, W_hh)
        print(f"  step {t}: {np.linalg.norm(dW_hh_step):.4f}")
    print("(Different steps contribute different amounts -- the shared weight's")
    print(" total gradient really is a sum over distinct per-step contributions.)\n")


def demo_truncated_bptt():
    rng = np.random.default_rng(2)
    input_size, hidden_size = 2, 3
    W_xh = rng.normal(size=(hidden_size, input_size)).astype(np.float32) * 0.4
    W_hh = rng.normal(size=(hidden_size, hidden_size)).astype(np.float32) * 0.4
    b_h = np.zeros(hidden_size, dtype=np.float32)

    full_sequence = [rng.normal(size=input_size).astype(np.float32) for _ in range(8)]
    dLoss_dh_final = rng.normal(size=hidden_size).astype(np.float32)

    # Full BPTT over all 8 steps
    full_dW_xh, full_dW_hh, full_db_h = bptt(full_sequence, W_xh, W_hh, b_h, dLoss_dh_final)

    # Truncated: split into two 4-step chunks, backprop only within each chunk
    chunk1, chunk2 = full_sequence[:4], full_sequence[4:]

    # Forward through chunk 1 to get the carried-over hidden state (no gradient tracked across chunks)
    h_mid = np.zeros(hidden_size, dtype=np.float32)
    for x_t in chunk1:
        h_mid, _ = rnn_cell_forward(x_t, h_mid, W_xh, W_hh, b_h)

    # Backprop only within chunk 2 (gradient does NOT flow back into chunk 1)
    trunc_dW_xh_2, trunc_dW_hh_2, trunc_db_h_2 = bptt(chunk2, W_xh, W_hh, b_h, dLoss_dh_final)

    print("Full BPTT dW_hh (all 8 steps):        norm =", np.linalg.norm(full_dW_hh).round(4))
    print("Truncated BPTT dW_hh (chunk 2 only):   norm =", np.linalg.norm(trunc_dW_hh_2).round(4))
    print("\nTruncated BPTT's gradient differs from the full computation -- it never sees")
    print("chunk 1's contribution at all, illustrating the tradeoff: a cheaper backward")
    print("pass, but dependencies spanning the chunk boundary can't be learned from.")


if __name__ == "__main__":
    print("=== BPTT vs PyTorch autograd ===")
    demo_bptt_vs_autograd()

    print("=== Per-step gradient contributions ===")
    demo_per_step_contributions()

    print("=== Truncated BPTT ===")
    demo_truncated_bptt()
