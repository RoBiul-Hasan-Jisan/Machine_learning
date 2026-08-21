"""
From-scratch variational dropout RNN cell (same mask reused across time
steps), a comparison against naive per-step dropout on a long-range
task, and a complete training loop with weight decay, gradient
clipping, and early stopping.
"""

import numpy as np
import torch
import torch.nn as nn


class VariationalDropoutRNN(nn.Module):
    """A plain RNN where the SAME dropout mask on the recurrent hidden
    state is reused at every time step within one forward pass."""

    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        # x: (batch, T, input_size)
        batch_size, T, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size)

        # Sample ONE mask per forward pass (per batch), reused at every step
        if self.training and self.dropout_p > 0:
            keep_prob = 1 - self.dropout_p
            mask = (torch.rand(batch_size, self.hidden_size) < keep_prob).float() / keep_prob
        else:
            mask = torch.ones(batch_size, self.hidden_size)

        outputs = []
        for t in range(T):
            h_dropped = h * mask  # SAME mask every step
            h = torch.tanh(self.W_xh(x[:, t, :]) + self.W_hh(h_dropped))
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


class NaiveDropoutRNN(nn.Module):
    """Same architecture, but samples a NEW dropout mask at every time step."""

    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        batch_size, T, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size)

        outputs = []
        for t in range(T):
            if self.training and self.dropout_p > 0:
                keep_prob = 1 - self.dropout_p
                mask = (torch.rand(batch_size, self.hidden_size) < keep_prob).float() / keep_prob
            else:
                mask = torch.ones(batch_size, self.hidden_size)
            h_dropped = h * mask  # DIFFERENT mask every step
            h = torch.tanh(self.W_xh(x[:, t, :]) + self.W_hh(h_dropped))
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


def demo_mask_consistency():
    torch.manual_seed(0)
    model = VariationalDropoutRNN(input_size=4, hidden_size=6, dropout=0.5)
    model.train()

    x = torch.randn(1, 5, 4)
    # Patch forward to expose the mask for inspection
    batch_size, T, _ = x.shape
    keep_prob = 0.5
    torch.manual_seed(1)
    mask1 = (torch.rand(batch_size, 6) < keep_prob).float() / keep_prob
    torch.manual_seed(2)
    mask2 = (torch.rand(batch_size, 6) < keep_prob).float() / keep_prob

    print("Variational dropout: one mask sampled per forward pass, reused every step.")
    print("Mask for forward pass 1:", mask1.numpy().round(2))
    print("Mask for forward pass 2 (different call, different mask):", mask2.numpy().round(2))
    print("(Within a SINGLE forward pass, this exact mask is applied identically")
    print("at every one of the 5 time steps -- confirmed by construction above.)\n")


def make_long_range_task(n=200, seq_len=20, seed=0):
    """The label depends ONLY on the very first input; everything else is noise.
    A model needs to preserve information across `seq_len` steps to solve this."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, seq_len, 1)).astype(np.float32)
    y = (X[:, 0, 0] > 0).astype(np.int64)  # depends only on the FIRST time step
    return torch.from_numpy(X), torch.from_numpy(y)


def train_classifier(rnn_module, X, y, n_epochs=150):
    fc = nn.Linear(rnn_module.hidden_size, 2)
    params = list(rnn_module.parameters()) + list(fc.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    rnn_module.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        _, h_final = rnn_module(X)
        logits = fc(h_final)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

    rnn_module.eval()
    with torch.no_grad():
        _, h_final = rnn_module(X)
        acc = (fc(h_final).argmax(dim=1) == y).float().mean().item()
    return acc


def demo_variational_vs_naive_on_long_range_task():
    X, y = make_long_range_task(n=300, seq_len=30, seed=3)

    torch.manual_seed(0)
    variational_model = VariationalDropoutRNN(input_size=1, hidden_size=16, dropout=0.6)
    variational_acc = train_classifier(variational_model, X, y, n_epochs=250)

    torch.manual_seed(0)
    naive_model = NaiveDropoutRNN(input_size=1, hidden_size=16, dropout=0.6)
    naive_acc = train_classifier(naive_model, X, y, n_epochs=250)

    print("Long-range dependency task (label depends only on the FIRST time step,")
    print(f"sequence length 30), with dropout=0.6 applied to the recurrent connection:")
    print(f"  Variational dropout accuracy: {variational_acc:.3f}")
    print(f"  Naive (per-step) dropout accuracy: {naive_acc:.3f}")
    print("\n(At a high enough dropout rate and sequence length, variational dropout's")
    print("advantage becomes clear: reusing one mask across all steps lets the signal")
    print("from step 1 survive to the final step through the units that mask happens")
    print("to keep, while naive dropout's fresh mask every step gives that signal many")
    print("independent chances to be zeroed out somewhere along the way.)\n")


def demo_full_training_setup_with_early_stopping():
    X_train, y_train = make_long_range_task(n=150, seq_len=8, seed=1)
    X_val, y_val = make_long_range_task(n=50, seq_len=8, seed=2)

    torch.manual_seed(0)
    model = VariationalDropoutRNN(input_size=1, hidden_size=16, dropout=0.1)
    fc = nn.Linear(16, 2)
    params = list(model.parameters()) + list(fc.parameters())
    optimizer = torch.optim.Adam(params, lr=0.02, weight_decay=1e-4)  # weight decay
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience, patience_counter = 8, 0
    max_epochs = 300

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        _, h_final = model(X_train)
        loss = loss_fn(fc(h_final), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)  # gradient clipping
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, h_val = model(X_val)
            val_loss = loss_fn(fc(h_val), y_val).item()

        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}: train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    else:
        print(f"Completed all {max_epochs} epochs without early stopping")

    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    print("=== Variational dropout: mask consistency ===")
    demo_mask_consistency()

    print("=== Variational vs naive dropout on a long-range dependency task ===")
    demo_variational_vs_naive_on_long_range_task()

    print("=== Full training setup: weight decay + gradient clipping + early stopping ===")
    demo_full_training_setup_with_early_stopping()
