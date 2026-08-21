# 02 · Training Neural Networks

Lesson 01 showed you how to compute *one* gradient step. This lesson is about the practical decisions that turn that single step into an actual training loop: how much data to look at before each update, how big a step to take, and how to change that step size over time.

## What you'll learn
- **Batch Gradient Descent** — compute the gradient using the *entire* dataset before each update
- **Stochastic Gradient Descent (SGD)** — compute the gradient using *one* example at a time
- **Mini-Batch SGD** — the practical middle ground everyone actually uses
- **Learning rate** — the single most important hyperparameter in deep learning
- **Epoch, batch, iteration** — the vocabulary of a training loop
- **Learning-rate scheduling** — changing the learning rate over the course of training

---

## 1. Batch Gradient Descent

Computes the gradient of the loss using **every** training example, then takes one step:

```
for epoch in range(num_epochs):
    gradient = compute_gradient(loss, ALL_training_data)
    W = W - lr * gradient
```

**Pros:** stable, smooth convergence — the gradient is the true gradient of the full loss surface.
**Cons:** for large datasets, one weight update requires a full pass over all the data — extremely slow, and it doesn't fit in memory once your dataset is large (e.g. millions of images).

## 2. Stochastic Gradient Descent (SGD)

Computes the gradient using **one randomly chosen example** at a time, and updates immediately:

```
for epoch in range(num_epochs):
    shuffle(data)
    for example in data:
        gradient = compute_gradient(loss, example)
        W = W - lr * gradient
```

**Pros:** extremely fast per-update, and the noise in the gradient estimate can actually help escape shallow local minima/saddle points.
**Cons:** the loss curve is very noisy (each step is a poor approximation of the true gradient); can't exploit vectorized/parallel hardware efficiently since you're only processing one example at a time.

## 3. Mini-Batch SGD

The practical compromise, and what virtually every modern deep learning system uses: compute the gradient over a **small batch** of examples (e.g. 32, 64, 128, 256) at a time.

```
for epoch in range(num_epochs):
    shuffle(data)
    for batch in split_into_batches(data, batch_size):
        gradient = compute_gradient(loss, batch)
        W = W - lr * gradient
```

**Why it wins:** batches are large enough to give a much smoother, less noisy gradient estimate than pure SGD, but small enough to (a) fit in GPU memory and (b) let you take many more update steps per epoch than full-batch GD, while still exploiting vectorized matrix operations for speed. It also retains some of SGD's helpful noise, which acts as a mild regularizer.

## 4. Learning Rate

The learning rate `η` scales the size of every gradient descent step: `W := W - η · ∂L/∂W`.

- **Too high:** the loss oscillates wildly or diverges — you overshoot the minimum every step.
- **Too low:** training crawls; you might run out of patience (or epochs) before converging, and can get stuck in shallow local minima or plateaus.
- **Just right:** loss decreases smoothly and reasonably quickly.

In practice, the learning rate is usually the single hyperparameter worth tuning first and most carefully — a good architecture with a bad learning rate will often train worse than a mediocre architecture with a good one.

## 5. Epoch, Batch, Iteration

These three terms are frequently confused — precise definitions:

- **Batch (size)** — the number of training examples used in *one* gradient update.
- **Iteration** — *one* gradient update (one pass through one batch).
- **Epoch** — *one* full pass through the *entire* training dataset.

If you have 10,000 training examples and a batch size of 100:
```
iterations per epoch = 10,000 / 100 = 100 iterations
```
So training for 20 epochs means 20 × 100 = 2,000 total weight updates.

## 6. Learning-Rate Scheduling

A fixed learning rate for the entire training run is rarely optimal: a larger rate helps early (fast initial progress), while a smaller rate helps later (fine-tuning near the minimum without overshooting it). A **scheduler** changes `η` over time according to a rule:

- **Step decay** — drop `η` by a factor (e.g. ×0.1) every N epochs
- **Exponential decay** — `η(t) = η₀ · e^(-kt)`, smoothly shrinking every step
- **Cosine annealing** — `η` follows a cosine curve down to (near) zero over training, popular in modern training recipes
- **Warmup** — start with a *small* `η` and ramp it *up* for the first few hundred/thousand steps before applying the main schedule; helps stabilize training at the very start when weights are randomly initialized and gradients can be unreliable
- **ReduceLROnPlateau** — automatically shrink `η` when a monitored metric (e.g. validation loss) stops improving for a set number of epochs

## Run the code
[`02-training-neural-networks.ipynb`] — implements Batch GD, SGD, and Mini-Batch SGD from scratch with NumPy on the same network from lesson 01, compares their loss curves, then reproduces the comparison in PyTorch using `DataLoader` and `torch.optim.lr_scheduler`.

