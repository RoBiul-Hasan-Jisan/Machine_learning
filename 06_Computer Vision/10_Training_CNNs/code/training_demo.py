"""
A complete training loop on a small CNN: SGD vs Adam convergence
comparison, plus BatchNorm and Dropout demonstrated in a real forward pass.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_toy_dataset(n=200, seed=0):
    """Two classes: images with a bright spot in the top-left vs bottom-right."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, 16, 16), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        label = rng.integers(0, 2)
        y[i] = label
        if label == 0:
            r, c = rng.integers(0, 5, size=2)
        else:
            r, c = rng.integers(11, 16, size=2)
        X[i, 0, r, c] = 1.0
        X[i, 0] += rng.normal(0, 0.05, size=(16, 16)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


class ToyCNN(nn.Module):
    def __init__(self, use_batchnorm=False, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8) if use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc = nn.Linear(8 * 8 * 8, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def train(model, X, y, optimizer, n_epochs=30):
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def demo_sgd_vs_adam():
    X, y = make_toy_dataset()

    torch.manual_seed(0)
    model_sgd = ToyCNN()
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
    losses_sgd = train(model_sgd, X, y, optimizer_sgd, n_epochs=30)

    torch.manual_seed(0)
    model_adam = ToyCNN()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.01)
    losses_adam = train(model_adam, X, y, optimizer_adam, n_epochs=30)

    print("Epoch  |  SGD loss  |  Adam loss")
    for i in [0, 4, 9, 19, 29]:
        print(f"{i + 1:5d}  |  {losses_sgd[i]:.4f}   |  {losses_adam[i]:.4f}")

    print(f"\nFinal SGD loss:  {losses_sgd[-1]:.4f}")
    print(f"Final Adam loss: {losses_adam[-1]:.4f}")
    print("(Adam typically converges faster on the same number of epochs / learning rate.)\n")


def demo_batchnorm_effect():
    X, y = make_toy_dataset()

    torch.manual_seed(1)
    model_no_bn = ToyCNN(use_batchnorm=False)
    optimizer = torch.optim.SGD(model_no_bn.parameters(), lr=0.1)  # deliberately high LR
    losses_no_bn = train(model_no_bn, X, y, optimizer, n_epochs=15)

    torch.manual_seed(1)
    model_bn = ToyCNN(use_batchnorm=True)
    optimizer = torch.optim.SGD(model_bn.parameters(), lr=0.1)
    losses_bn = train(model_bn, X, y, optimizer, n_epochs=15)

    print("With a deliberately high learning rate:")
    print(f"Final loss WITHOUT BatchNorm: {losses_no_bn[-1]:.4f}")
    print(f"Final loss WITH BatchNorm:    {losses_bn[-1]:.4f}")
    print("(BatchNorm often tolerates a higher learning rate more gracefully.)\n")


def demo_dropout_train_vs_eval():
    model = ToyCNN(dropout_p=0.5)
    x = torch.randn(1, 1, 16, 16)

    model.train()
    out_train_1 = model(x)
    out_train_2 = model(x)

    model.eval()
    out_eval_1 = model(x)
    out_eval_2 = model(x)

    print("In train mode, dropout is stochastic -> repeated calls differ:")
    print("Difference between two train-mode calls:", (out_train_1 - out_train_2).abs().sum().item())

    print("\nIn eval mode, dropout is disabled -> repeated calls are identical:")
    print("Difference between two eval-mode calls:", (out_eval_1 - out_eval_2).abs().sum().item())
    assert torch.allclose(out_eval_1, out_eval_2)
    print("Confirmed: eval-mode outputs are deterministic.")


if __name__ == "__main__":
    print("=== SGD vs Adam convergence ===")
    demo_sgd_vs_adam()

    print("=== BatchNorm effect at a high learning rate ===")
    demo_batchnorm_effect()

    print("=== Dropout: train mode (stochastic) vs eval mode (deterministic) ===")
    demo_dropout_train_vs_eval()
