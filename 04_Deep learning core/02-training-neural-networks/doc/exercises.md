# Exercises — 02 Training Neural Networks

1. **Vocabulary check.** You have 50,000 training images, a batch size of 250, and you train for 15 epochs. How many iterations per epoch? How many total iterations?
2. **Noise vs speed.** Rerun the NumPy comparison cell with `batch_size=1` (SGD) and `batch_size=len(X_train)` (Batch GD) for only 20 epochs. Which one has made more visible progress in that time, and why — think about how many *weight updates* each has performed, not just epochs.
3. **Too-high learning rate.** In the PyTorch `run_training` function, set `lr=20` for the mini-batch run. What happens to the loss curve? Explain what's happening geometrically on the loss surface.
4. **Design a schedule.** Using `step_decay` or `cosine_annealing` as a template, write a "warmup + cosine decay" schedule function: learning rate linearly increases from 0 to `initial_lr` over the first 10 epochs, then follows cosine annealing for the rest. Plot it.
5. **Batch size trade-off.** Explain, using what you know about mini-batch SGD, why doubling the batch size (e.g. 32 → 64) usually lets you *also* increase the learning rate somewhat, without destabilizing training.
