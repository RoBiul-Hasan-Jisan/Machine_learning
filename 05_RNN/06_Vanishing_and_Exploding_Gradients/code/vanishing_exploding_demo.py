"""
Empirical demonstration of vanishing and exploding gradients in a plain
RNN as sequence length grows, plus gradient clipping in action.
"""

import numpy as np
import torch
import torch.nn as nn


def gradient_norm_at_input(seq_len, hidden_size=8, input_size=4, W_hh_scale=1.0, seed=0):
    torch.manual_seed(seed)
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    with torch.no_grad():
        rnn.weight_hh_l0.mul_(W_hh_scale)

    x = torch.randn(1, seq_len, input_size, requires_grad=True)
    _, h_n = rnn(x)
    loss = h_n.sum()
    loss.backward()

    # Gradient reaching the very first time step's input
    grad_at_first_step = x.grad[0, 0].norm().item()
    return grad_at_first_step


def demo_vanishing_gradients():
    print("W_hh scaled DOWN (0.3x) -- gradient at the first time step, by sequence length:")
    for seq_len in [5, 20, 50, 100]:
        grad_norm = gradient_norm_at_input(seq_len, W_hh_scale=0.3)
        print(f"  seq_len={seq_len:4d}: grad norm at step 0 = {grad_norm:.8f}")
    print("(Shrinks toward zero as sequence length grows -- vanishing gradient.)\n")


def demo_exploding_gradients():
    print("W_hh scaled UP (1.8x) -- gradient at the first time step, by sequence length:")
    for seq_len in [3, 6, 9, 12]:
        grad_norm = gradient_norm_at_input(seq_len, W_hh_scale=1.8)
        print(f"  seq_len={seq_len:4d}: grad norm at step 0 = {grad_norm:.4f}")
    print("(With a real tanh RNN, growth from the >1 weight scale competes with")
    print(" tanh's saturation -- once pre-activations get large, tanh'(z) shrinks")
    print(" toward zero and can mask or reverse the raw exploding-weight effect.")
    print(" The idealized linear picture below isolates the effect the lesson's")
    print(" math describes, without that complication.)\n")


def demo_exploding_gradients_idealized():
    """Isolate the pure linear-algebra effect (no activation function) by
    tracking ||W_hh^k|| as k grows -- this is exactly the repeated-matrix-
    multiplication term from the lesson's derivation, with tanh's saturation
    removed from the picture."""
    rng = np.random.default_rng(0)
    hidden_size = 8

    W_shrink = rng.normal(size=(hidden_size, hidden_size)).astype(np.float64) * 0.05
    W_grow = rng.normal(size=(hidden_size, hidden_size)).astype(np.float64) * 0.5

    print("||W_hh^k|| (spectral norm) as k grows -- shrinking weight matrix:")
    for k in [1, 5, 10, 20]:
        norm = np.linalg.norm(np.linalg.matrix_power(W_shrink, k), ord=2)
        print(f"  k={k:3d}: ||W_hh^k|| = {norm:.2e}")

    print("\n||W_hh^k|| (spectral norm) as k grows -- growing weight matrix:")
    for k in [1, 5, 10, 20]:
        norm = np.linalg.norm(np.linalg.matrix_power(W_grow, k), ord=2)
        print(f"  k={k:3d}: ||W_hh^k|| = {norm:.2e}")
    print("\nThis is the idealized version of the repeated-multiplication term from")
    print("the lesson's derivation: shrinks toward 0 or grows without bound purely")
    print("from the matrix's spectral radius, before any activation function is applied.\n")


def demo_gradient_clipping():
    torch.manual_seed(0)
    hidden_size, input_size, seq_len = 8, 4, 30

    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    with torch.no_grad():
        rnn.weight_hh_l0.mul_(2.5)  # deliberately unstable scale

    x = torch.randn(1, seq_len, input_size)
    _, h_n = rnn(x)
    loss = h_n.sum()
    loss.backward()

    total_norm_before = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in rnn.parameters() if p.grad is not None)
    ).item()

    # Manual clipping implementation
    max_norm = 5.0
    total_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in rnn.parameters() if p.grad is not None)
    )
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in rnn.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    total_norm_after_manual = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in rnn.parameters() if p.grad is not None)
    ).item()

    print(f"Total gradient norm before clipping: {total_norm_before:.4f}")
    print(f"Total gradient norm after MANUAL clipping (max_norm=5.0): {total_norm_after_manual:.4f}")

    # Compare against PyTorch's built-in clipping on a fresh copy
    torch.manual_seed(0)
    rnn2 = nn.RNN(input_size, hidden_size, batch_first=True)
    with torch.no_grad():
        rnn2.weight_hh_l0.mul_(2.5)
    x2 = torch.randn(1, seq_len, input_size)
    _, h_n2 = rnn2(x2)
    loss2 = h_n2.sum()
    loss2.backward()
    torch.nn.utils.clip_grad_norm_(rnn2.parameters(), max_norm=5.0)
    total_norm_builtin = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in rnn2.parameters() if p.grad is not None)
    ).item()

    print(f"Total gradient norm after BUILT-IN clip_grad_norm_:        {total_norm_builtin:.4f}")
    print(f"\nBoth capped at (approximately) the threshold of {max_norm}, confirming")
    print("the manual implementation matches PyTorch's built-in clipping.")


if __name__ == "__main__":
    print("=== Vanishing gradients ===")
    demo_vanishing_gradients()

    print("=== Exploding gradients (real tanh RNN) ===")
    demo_exploding_gradients()

    print("=== Exploding/vanishing, idealized (pure linear algebra) ===")
    demo_exploding_gradients_idealized()

    print("=== Gradient clipping ===")
    demo_gradient_clipping()
