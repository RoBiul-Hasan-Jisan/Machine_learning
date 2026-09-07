"""
gd_utils.py
-----------
Shared, vectorized building blocks for the gradient-descent-variant demos.

This module replaces the copy-pasted helper functions that used to live at
the top of every notebook. Consolidating them here means:

  * one correct implementation instead of eight near-duplicates
  * fully vectorized math (no Python-level ``for`` loops over data points)
  * a single, well-tested ``gradient_descent`` function that supports every
    variant (vanilla, decaying LR, line search, momentum, Nesterov, Adagrad,
    RMSProp, Adam) through one consistent, keyword-driven interface

Model
-----
We fit a single-input logistic (sigmoid) unit  f(x) = sigmoid(w*x + b)
to data (x, y) using either mean-squared error or binary cross-entropy.

Bugs fixed relative to the original notebooks
----------------------------------------------
1. ``b`` update in momentum/Nesterov used the *weight* update instead of the
   *bias* update (``b = b - updated_w``). Fixed to use ``updated_b``.
2. Cross-entropy loss/gradient only included the positive-class term
   (``-y*log(f)``), which is not a valid loss. Now uses the full binary
   cross-entropy ``-(y*log(f) + (1-y)*log(1-f))``, whose gradient w.r.t.
   the sigmoid pre-activation has the clean closed form ``(f - y)``.
3. Line-search reused the loop variable ``i`` for both the epoch/sample
   index *and* the candidate-learning-rate index, and overwrote the string
   ``loss`` argument with a numeric loss value inside the loop -- breaking
   every ``loss == 'mse'`` check after the first mini-batch. Fixed by using
   distinct variable names.
4. ``np.float`` (removed in NumPy >= 1.24) replaced with plain ``float``.
5. Manual point-counter (``points % batch_size``) mini-batching replaced
   with clean slicing over shuffled indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np



# Core math (all vectorized: these operate on whole arrays, not scalars)


def sigmoid(x, w, b):
    """Sigmoid activation f(x) = 1 / (1 + exp(-(w*x + b)))."""
    return 1.0 / (1.0 + np.exp(-(w * x + b)))


def mse_loss(x, y, w, b):
    """Mean squared error between y and sigmoid(w*x+b)."""
    fx = sigmoid(x, w, b)
    return float(np.mean(0.5 * (y - fx) ** 2))


def bce_loss(x, y, w, b, eps=1e-12):
    """Binary cross-entropy loss (both classes, numerically clipped)."""
    fx = np.clip(sigmoid(x, w, b), eps, 1 - eps)
    return float(np.mean(-(y * np.log(fx) + (1 - y) * np.log(1 - fx))))


def mse_grad(x, y, w, b):
    """Gradient of mse_loss w.r.t. (w, b), vectorized + averaged."""
    fx = sigmoid(x, w, b)
    err = (fx - y) * fx * (1 - fx)
    dw = np.mean(err * x)
    db = np.mean(err)
    return dw, db


def bce_grad(x, y, w, b):
    """Gradient of bce_loss w.r.t. (w, b). Simplifies to (f - y) for a
    sigmoid unit trained with cross-entropy."""
    fx = sigmoid(x, w, b)
    err = fx - y
    dw = np.mean(err * x)
    db = np.mean(err)
    return dw, db


LOSS_FNS = {
    "mse": (mse_loss, mse_grad),
    "cross_entropy": (bce_loss, bce_grad),
}



# Data loading


def load_xy(path="sgd_data.txt"):
    """Load two-column (x, y) CSV data as flat 1-D float arrays."""
    data = np.genfromtxt(path, delimiter=",")
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    return x, y



# Result container


@dataclass
class GDResult:
    w_history: list = field(default_factory=list)
    b_history: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    w: float = 0.0
    b: float = 0.0
    n_iterations: int = 0  # only meaningful for the 'decay' variant



# Unified gradient descent driver


def gradient_descent(
    x,
    y,
    epochs=500,
    batch_size=10,
    loss="mse",
    variant="vanilla",
    lr=0.01,
    eta=0.1,
    beta=0.95,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    lr_candidates=None,
    seed=None,
    verbose_every=50,
):
    """Run one of several mini-batch gradient descent variants.

    Parameters
    ----------
    x, y : 1-D arrays of training data.
    epochs : number of passes over the data (the 'decay' variant instead
        counts *successful* epochs and may run more iterations than this).
    batch_size : mini-batch size.
    loss : 'mse' or 'cross_entropy'.
    variant : one of 'vanilla', 'decay', 'line_search', 'momentum',
        'nesterov', 'adagrad', 'rmsprop', 'adam'.
    lr : (initial) learning rate. Ignored by 'line_search' in favor of
        ``lr_candidates``.
    eta : step-size scale used by 'momentum' and 'nesterov'.
    beta : decay rate for 'rmsprop'.
    beta1, beta2 : decay rates for 'adam'.
    epsilon : numerical-stability constant for the adaptive methods.
    lr_candidates : list of learning rates tried at every step by
        'line_search' (defaults to a small preset list).
    seed : optional RNG seed for reproducibility.
    verbose_every : print the loss every N epochs (0 disables printing).

    Returns
    -------
    GDResult
    """
    if loss not in LOSS_FNS:
        raise ValueError(f"loss must be one of {list(LOSS_FNS)}, got {loss!r}")
    loss_fn, grad_fn = LOSS_FNS[loss]

    rng = np.random.default_rng(seed)
    w = float(rng.standard_normal())
    b = float(rng.standard_normal())

    n = x.shape[0]
    result = GDResult()

    # Per-variant state
    momentum_w = momentum_b = 0.0          # momentum / nesterov / adam
    update_w = update_b = 0.0              # adagrad / rmsprop / adam
    if lr_candidates is None:
        lr_candidates = [0.01, 0.07, 0.1, 0.2, 0.4, 0.9]

    def batches():
        """Yield shuffled mini-batches of (x, y) covering one epoch."""
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b_idx = idx[start:start + batch_size]
            yield x[b_idx], y[b_idx]

    def log_epoch(epoch_idx, w, b):
        cur_loss = loss_fn(x, y, w, b)
        result.loss_history.append(cur_loss)
        result.w_history.append(w)
        result.b_history.append(b)
        if verbose_every and epoch_idx % verbose_every == 0:
            print(f"[{variant}] epoch {epoch_idx:4d}  loss = {cur_loss:.6f}")
        return cur_loss

    # ---- 'decay' variant has its own control flow (retries an epoch on a
    # loss increase, instead of just moving on) ----
    if variant == "decay":
        epoch = 0
        iterations = 0
        prev_loss = np.inf
        while epoch <= epochs:
            iterations += 1
            w_before, b_before = w, b
            for xb, yb in batches():
                dw, db = grad_fn(xb, yb, w, b)
                w -= lr * dw
                b -= lr * db
            cur_loss = loss_fn(x, y, w, b)
            if cur_loss < prev_loss:
                prev_loss = cur_loss
                log_epoch(epoch, w, b)
                epoch += 1
            else:
                lr /= 2.0
                w, b = w_before, b_before
        result.n_iterations = iterations
        result.w, result.b = w, b
        return result

    # ---- all other variants share one simple loop ----
    for epoch in range(epochs + 1):
        for xb, yb in batches():
            dw, db = grad_fn(xb, yb, w, b)

            if variant == "vanilla":
                w -= lr * dw
                b -= lr * db

            elif variant == "line_search":
                best_w, best_b, best_loss = w, b, np.inf
                for cand_lr in lr_candidates:
                    trial_w = w - cand_lr * dw
                    trial_b = b - cand_lr * db
                    trial_loss = loss_fn(x, y, trial_w, trial_b)
                    if trial_loss < best_loss:
                        best_loss, best_w, best_b = trial_loss, trial_w, trial_b
                w, b = best_w, best_b

            elif variant == "momentum":
                update_w = lr * momentum_w + eta * dw
                update_b = lr * momentum_b + eta * db
                w -= update_w
                b -= update_b
                momentum_w, momentum_b = update_w, update_b

            elif variant == "nesterov":
                # Gradient is evaluated at the *look-ahead* point.
                look_w = w - lr * momentum_w
                look_b = b - lr * momentum_b
                dw_la, db_la = grad_fn(xb, yb, look_w, look_b)
                update_w = lr * momentum_w + eta * dw_la
                update_b = lr * momentum_b + eta * db_la
                w -= update_w
                b -= update_b
                momentum_w, momentum_b = update_w, update_b

            elif variant == "adagrad":
                update_w += dw ** 2
                update_b += db ** 2
                w -= (lr / np.sqrt(update_w + epsilon)) * dw
                b -= (lr / np.sqrt(update_b + epsilon)) * db

            elif variant == "rmsprop":
                update_w = beta * update_w + (1 - beta) * dw ** 2
                update_b = beta * update_b + (1 - beta) * db ** 2
                w -= (lr / np.sqrt(update_w + epsilon)) * dw
                b -= (lr / np.sqrt(update_b + epsilon)) * db

            elif variant == "adam":
                momentum_w = beta1 * momentum_w + (1 - beta1) * dw
                momentum_b = beta1 * momentum_b + (1 - beta1) * db
                update_w = beta2 * update_w + (1 - beta2) * dw ** 2
                update_b = beta2 * update_b + (1 - beta2) * db ** 2
                mw_hat = momentum_w / (1 - beta1 ** (epoch + 1))
                mb_hat = momentum_b / (1 - beta1 ** (epoch + 1))
                uw_hat = update_w / (1 - beta2 ** (epoch + 1))
                ub_hat = update_b / (1 - beta2 ** (epoch + 1))
                w -= (lr / np.sqrt(uw_hat + epsilon)) * mw_hat
                b -= (lr / np.sqrt(ub_hat + epsilon)) * mb_hat

            else:
                raise ValueError(f"Unknown variant: {variant!r}")

        log_epoch(epoch, w, b)

    result.w, result.b = w, b
    return result



# Plotting helpers


def plot_loss_curve(result: GDResult, title="Loss vs. Epoch"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(result.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_error_surface(x, y, loss="mse", w_range=(-10, 10), b_range=(-10, 10),
                        num=200, title=None):
    """Vectorized error-surface plot (no Python loop over the grid)."""
    import matplotlib.pyplot as plt

    loss_fn, _ = LOSS_FNS[loss]
    w_vals = np.linspace(*w_range, num=num)
    b_vals = np.linspace(*b_range, num=num)
    W, B = np.meshgrid(w_vals, b_vals)

    # Vectorized surface: broadcast x,y over the (num, num) grid.
    fx = sigmoid(x[None, None, :], W[:, :, None], B[:, :, None])
    if loss == "mse":
        surface = np.mean(0.5 * (y[None, None, :] - fx) ** 2, axis=-1)
    else:
        fx = np.clip(fx, 1e-12, 1 - 1e-12)
        surface = np.mean(
            -(y[None, None, :] * np.log(fx) + (1 - y[None, None, :]) * np.log(1 - fx)),
            axis=-1,
        )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(W, B, surface, cmap="coolwarm", linewidth=0, antialiased=False)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("loss")
    ax.set_title(title or f"{loss} error surface")
    plt.tight_layout()
    plt.show()
