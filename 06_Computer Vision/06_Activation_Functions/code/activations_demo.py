"""
ReLU, Leaky ReLU, and GELU compared numerically, plus a demonstration
of the dying ReLU problem and how Leaky ReLU mitigates it.
"""

import numpy as np
import torch
import torch.nn as nn


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def gelu(x):
    from scipy.stats import norm
    return x * norm.cdf(x)


def demo_activation_values():
    x = np.array([-3, -1, -0.1, 0, 0.1, 1, 3])
    print("x:            ", x)
    print("ReLU(x):      ", relu(x))
    print("LeakyReLU(x): ", leaky_relu(x).round(4))
    print("GELU(x):      ", gelu(x).round(4))


def demo_linear_collapse():
    """Show that stacking linear layers with no activation collapses to
    a single linear layer."""
    rng = np.random.default_rng(0)
    W1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4)
    W2, b2 = rng.normal(size=(2, 4)), rng.normal(size=2)

    x = rng.normal(size=3)

    # Two linear layers stacked
    stacked_output = W2 @ (W1 @ x + b1) + b2

    # Equivalent single linear layer
    W_combined = W2 @ W1
    b_combined = W2 @ b1 + b2
    single_output = W_combined @ x + b_combined

    print("\nStacked linear layers output:", stacked_output.round(6))
    print("Equivalent single linear layer output:", single_output.round(6))
    assert np.allclose(stacked_output, single_output)
    print("Identical — confirms two linear layers collapse to one without a nonlinearity.\n")


def demo_dying_relu():
    torch.manual_seed(0)

    # A single ReLU unit whose weights are pushed so its pre-activation
    # is negative for every input in this toy dataset.
    X = torch.tensor([[1.0], [2.0], [3.0], [-1.0], [-2.0]])
    weight = torch.tensor([[-5.0]], requires_grad=True)  # strongly negative weight
    bias = torch.tensor([-10.0], requires_grad=True)      # strongly negative bias

    pre_activation = X @ weight.T + bias
    relu_out = torch.relu(pre_activation)
    loss_relu = relu_out.sum()
    loss_relu.backward()
    relu_grad = weight.grad.clone()

    weight.grad = None
    pre_activation2 = X @ weight.T + bias  # recompute graph for second backward pass
    leaky_out = torch.nn.functional.leaky_relu(pre_activation2, negative_slope=0.01)
    loss_leaky = leaky_out.sum()
    loss_leaky.backward()
    leaky_grad = weight.grad.clone()

    print("Pre-activation values (all negative -> ReLU outputs all zero):")
    print(pre_activation.detach().numpy().ravel().round(2))
    print("ReLU gradient w.r.t. weight (dead unit -> should be 0):", relu_grad.item())
    print("Leaky ReLU gradient w.r.t. weight (should be nonzero):", leaky_grad.item())
    assert relu_grad.item() == 0.0
    assert leaky_grad.item() != 0.0
    print("Confirmed: ReLU's dead unit cannot learn (zero gradient),")
    print("while Leaky ReLU still receives a small gradient signal.")


if __name__ == "__main__":
    print("=== Activation function values ===")
    demo_activation_values()

    print("\n=== Linear layers collapse without activation ===")
    demo_linear_collapse()

    print("=== Dying ReLU demonstration ===")
    demo_dying_relu()
