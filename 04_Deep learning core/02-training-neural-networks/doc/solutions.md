# Solutions — 02 Training Neural Networks

1. Iterations per epoch = 50,000 / 250 = 200. Total iterations = 200 × 15 = 3,000.

2. In 20 epochs, Batch GD has performed only 20 weight updates total (one per epoch, since it uses the whole dataset each time), while SGD has performed `20 × len(X_train)` updates (one per example, per epoch) — likely thousands. Even though each individual SGD update is noisier/less accurate, the sheer number of updates usually means SGD's loss has dropped further in wall-clock/epoch terms — though its curve will look much choppier than Batch GD's smooth descent.

3. With `lr=20`, the loss curve typically explodes, oscillates wildly, or even becomes `NaN`. Geometrically, each gradient step is now far too large: instead of taking a small step downhill on the loss surface, the optimizer leaps clear over the minimum to the opposite (and often steeper) side of the loss "bowl," causing the loss to increase rather than decrease, and this compounds every subsequent step, causing divergence.

4. Example implementation:
```python
def warmup_cosine(epoch, initial_lr=1.0, warmup_epochs=10, total_epochs=100):
    if epoch < warmup_epochs:
        return initial_lr * (epoch / warmup_epochs)  # linear ramp-up from 0
    # cosine decay over the remaining epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
```
Plotting `[warmup_cosine(e) for e in range(100)]` should show a straight ramp from 0 to `initial_lr` over the first 10 epochs, followed by a smooth cosine-shaped decay back toward 0.

5. A larger batch size gives a gradient estimate that is a better (lower-variance) approximation of the true full-dataset gradient — the per-step noise is reduced roughly proportional to `1/sqrt(batch_size)`. Because the gradient estimate is more reliable/stable, you can afford to trust it more and take a bigger step (higher learning rate) without as much risk of that step being based on a wildly inaccurate, noisy gradient — this is the intuition behind the common heuristic of scaling the learning rate roughly linearly with batch size.
