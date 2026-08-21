"""
End-to-end RNN training loop: masked loss on padded sequences, gradient
clipping, truncated BPTT with .detach(), and a validation loop, on a
small synthetic sequence classification task.
"""

import numpy as np
import torch
import torch.nn as nn


def make_padded_dataset(n=200, max_len=12, min_len=3, num_classes=2, seed=0):
    """Sequences of random length; label depends on the sign of the sum
    of the (real, non-padded) values."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_len, max_len + 1, size=n)

    X = np.zeros((n, max_len, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.float32)

    for i in range(n):
        length = lengths[i]
        values = rng.normal(0, 1, size=length).astype(np.float32)
        X[i, :length, 0] = values
        mask[i, :length] = 1.0
        y[i] = int(values.sum() > 0)

    return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(lengths)


class SequenceClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_classes=2):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.rnn(x)  # output: (batch, T, hidden_size) -- every step
        logits_per_step = self.fc(output)  # (batch, T, num_classes) -- for masked per-step demo
        return logits_per_step


def masked_loss(logits, targets_per_step, mask, loss_fn):
    """logits: (batch, T, num_classes). targets_per_step: (batch, T).
    mask: (batch, T). Averages loss over only the real (non-padded) positions."""
    batch, T, num_classes = logits.shape
    losses = loss_fn(logits.reshape(-1, num_classes), targets_per_step.reshape(-1))  # (batch*T,)
    losses = losses * mask.reshape(-1)
    return losses.sum() / mask.sum()


def demo_masked_vs_unmasked_loss():
    X, y, mask, lengths = make_padded_dataset(n=8, max_len=12, min_len=3)
    model = SequenceClassifier()
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Broadcast the single per-sequence label to every time step, for this demo
    targets_per_step = y.unsqueeze(1).expand(-1, X.shape[1])

    logits = model(X)
    masked = masked_loss(logits, targets_per_step, mask, loss_fn)

    # Unmasked: naive average over ALL positions, including padding
    unmasked = loss_fn(logits.reshape(-1, 2), targets_per_step.reshape(-1)).mean()

    avg_real_frac = mask.mean().item()
    print(f"Average fraction of REAL (non-padded) positions in this batch: {avg_real_frac:.2f}")
    print(f"Masked loss (averaged over real positions only):   {masked.item():.4f}")
    print(f"Unmasked loss (naively averaged over ALL positions): {unmasked.item():.4f}")
    print("(These differ because the unmasked version lets padding positions -- which")
    print("carry no real signal -- dilute the loss and gradient.)\n")


def demo_gradient_clipping_prevents_nan():
    torch.manual_seed(0)
    input_size, hidden_size, T = 4, 8, 150

    for use_clipping in [False, True]:
        torch.manual_seed(0)
        rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="relu")
        with torch.no_grad():
            rnn.weight_hh_l0.copy_(torch.eye(hidden_size) * 1.15)  # deliberately unstable, per Lesson 06
        fc = nn.Linear(hidden_size, 2)
        params = list(rnn.parameters()) + list(fc.parameters())
        optimizer = torch.optim.SGD(params, lr=0.5)
        loss_fn = nn.CrossEntropyLoss()

        x = torch.randn(4, T, input_size)
        y = torch.randint(0, 2, (4,))

        went_nan = False
        max_grad_norm_seen = 0.0
        for _ in range(10):
            optimizer.zero_grad()
            output, h_n = rnn(x)
            logits = fc(h_n.squeeze(0))
            loss = loss_fn(logits, y)
            loss.backward()

            total_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)).item()
            max_grad_norm_seen = max(max_grad_norm_seen, total_norm)

            if use_clipping:
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                went_nan = True
                break

        label = "WITH clipping   " if use_clipping else "WITHOUT clipping"
        print(f"{label}: max gradient norm seen = {max_grad_norm_seen:.2e}   went NaN/Inf? {went_nan}")
    print("(An unclipped huge gradient norm, fed straight into an SGD update, is exactly")
    print(" what pushes weights into NaN territory; clipping caps the update size directly.)\n")


def demo_truncated_bptt_detach():
    torch.manual_seed(0)
    input_size, hidden_size = 4, 8
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    chunk1 = torch.randn(1, 10, input_size)
    chunk2 = torch.randn(1, 10, input_size)

    h = torch.zeros(1, 1, hidden_size)

    # Process chunk 1, keep gradient history
    out1, h = rnn(chunk1, h)
    print("Before detach: does h require grad tracking back through chunk 1?", h.requires_grad)

    h = h.detach()
    print("After detach:  does h require grad tracking back through chunk 1?", h.requires_grad)

    out2, h2 = rnn(chunk2, h)
    loss = out2.sum()
    loss.backward()  # this should NOT try to backprop into chunk1's graph (already detached)
    print("\nBackward pass through chunk 2 completed without needing chunk 1's graph --")
    print("confirms truncated BPTT: memory (h's VALUES) carried forward, but gradient")
    print("history cut at the chunk boundary.")


def demo_full_training_and_validation():
    torch.manual_seed(0)
    X_train, y_train, mask_train, _ = make_padded_dataset(n=150, seed=1)
    X_val, y_val, mask_val, _ = make_padded_dataset(n=50, seed=2)

    model = SequenceClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    targets_train = y_train.unsqueeze(1).expand(-1, X_train.shape[1])
    targets_val = y_val.unsqueeze(1).expand(-1, X_val.shape[1])

    print("\n=== Full training loop with masked loss + gradient clipping ===")
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = masked_loss(logits, targets_train, mask_train, loss_fn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % 3 == 0 or epoch == 14:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = masked_loss(val_logits, targets_val, mask_val, loss_fn)
                # accuracy at each sequence's LAST real position
                lengths_val = mask_val.sum(dim=1).long() - 1
                last_logits = val_logits[torch.arange(len(y_val)), lengths_val]
                val_acc = (last_logits.argmax(dim=1) == y_val).float().mean().item()
            print(f"Epoch {epoch + 1:2d}: train_loss={loss.item():.4f}  "
                  f"val_loss={val_loss.item():.4f}  val_acc={val_acc:.3f}")


if __name__ == "__main__":
    print("=== Masked vs unmasked loss ===")
    demo_masked_vs_unmasked_loss()

    print("=== Gradient clipping prevents NaN ===")
    demo_gradient_clipping_prevents_nan()

    print("\n=== Truncated BPTT with detach() ===")
    demo_truncated_bptt_detach()

    demo_full_training_and_validation()
