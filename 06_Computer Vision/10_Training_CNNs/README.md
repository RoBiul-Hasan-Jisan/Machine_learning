# 10. Training CNNs

## Learning Objectives

- Assemble forward propagation, loss computation, backpropagation, and an optimizer into a full training loop
- Compare SGD, SGD with momentum, and Adam, and know when to reach for each
- Apply the standard regularization and stabilization techniques: weight decay, dropout, batch normalization, and learning rate scheduling

## The Problem

Lessons 08-09 covered forward propagation and backpropagation for a single input. Training is running that cycle — forward pass, compute loss, backward pass, update weights — repeatedly over many batches and many passes through the dataset (epochs), while managing the practical issues that come with actually getting a CNN to converge to something useful: choosing a learning rate, preventing overfitting, keeping gradients well-behaved as depth increases.

## The Concept

### The training loop

```python
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()               # clear gradients from the previous step
        y_pred = model(X_batch)             # forward pass (Lesson 08)
        loss = loss_fn(y_pred, y_batch)      # e.g. cross-entropy loss
        loss.backward()                      # backward pass (Lesson 09) - autograd computes all gradients
        optimizer.step()                     # update every weight using its gradient
```

Every piece of this loop maps to something covered earlier in this module: `model(X_batch)` is Lesson 08's forward propagation; `loss.backward()` triggers exactly the chain-rule gradient computations derived in Lesson 09, automatically, for every layer.

### Loss functions for classification

**Cross-entropy loss** is the standard choice, pairing naturally with the softmax output from Lesson 08:

```
CrossEntropy(y_true, y_hat) = -log(y_hat[true_class])
```

It penalizes confident wrong predictions heavily (as `y_hat[true_class]` approaches 0, `-log(...)` approaches infinity) and barely penalizes confident correct predictions. In practice, `torch.nn.CrossEntropyLoss` combines softmax and cross-entropy into one numerically stable operation — you pass it raw logits, not post-softmax probabilities.

### Optimizers: how weights actually get updated

**SGD (Stochastic Gradient Descent)** — the baseline:

```
w = w - learning_rate * dLoss/dw
```

Simple, but can be slow to converge and sensitive to a poorly chosen learning rate — too high overshoots and diverges, too low takes forever.

**SGD with momentum** — accumulates a running average of past gradients, which smooths out noisy updates and helps push through shallow regions of the loss surface:

```
velocity = momentum * velocity - learning_rate * dLoss/dw
w = w + velocity
```

**Adam** — adapts the effective learning rate per parameter based on recent gradient magnitude and variance, combining momentum with a per-parameter learning rate adjustment. It's the most common default for training from scratch because it typically converges faster and is less sensitive to the initial learning rate choice than plain SGD.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# vs
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

A common pattern in practice: use Adam for fast initial convergence, or use well-tuned SGD with momentum for the final, best generalization performance — several landmark architectures (Lessons 11-16) were originally trained with SGD and careful learning rate schedules.

### Regularization: preventing overfitting

**Weight decay (L2 regularization)** adds a penalty proportional to the squared weights to the loss, discouraging any single weight from growing unnecessarily large — the same idea as Ridge regression's penalty, applied to a neural network's weights:

```
Loss_total = Loss_task + weight_decay * sum(w^2 for all weights w)
```

**Dropout** randomly zeroes a fraction of units during training (never at inference time), forcing the network to not rely too heavily on any single unit or co-adapted group of units:

```python
self.dropout = nn.Dropout(p=0.5)   # 50% of units zeroed during training
```

Dropout is typically applied to fully connected layers near the end of a CNN, less commonly within convolutional layers (batch normalization, below, tends to serve a similar stabilizing role there).

**Batch normalization** normalizes each layer's activations (zero mean, unit variance) across the current batch, then applies a small learnable scale and shift. This stabilizes and speeds up training by keeping activation distributions consistent as weights change during training, and has a mild regularizing side effect from the noise introduced by per-batch statistics.

```python
self.bn1 = nn.BatchNorm2d(num_features=64)   # applied after conv1, before activation
```

The typical order is `Conv → BatchNorm → Activation → Pool`, batch norm sitting right after the linear conv operation and before the nonlinearity.

### Learning rate scheduling

Keeping the learning rate fixed for the whole training run is rarely optimal: a larger rate helps early on for fast progress, a smaller rate helps later for fine-tuning without overshooting a good solution.

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# reduces the learning rate by 10x every 10 epochs

for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()
```

### Monitoring training: loss curves and overfitting

Track training loss and validation loss (Lesson 02's split methodology, unchanged for deep learning) across epochs:

```
Training loss keeps decreasing, validation loss starts increasing  →  overfitting
Both training and validation loss plateau at a high value           →  underfitting
                                                                          (increase capacity, train longer, or tune LR)
```

This is the same diagnostic logic from the classical ML lessons — the failure modes are conceptually identical, and the fixes (regularization, more data, simpler model for overfitting; bigger model, more training, better learning rate for underfitting) carry over directly.

See `code/training_demo.py` for a complete training loop on a small CNN, comparing SGD vs Adam convergence speed, and demonstrating batch normalization and dropout in a real forward pass.

## Exercises

1. Train the `SimpleCNN` from Lesson 07 on a small synthetic dataset with SGD (no momentum) vs Adam, using the same number of epochs. Plot training loss for both and compare convergence speed.
2. Add `BatchNorm2d` after each convolutional layer in a small CNN and compare training loss curves with and without it, using the same learning rate.
3. Add `Dropout(p=0.5)` before the final FC layer, and compare training accuracy vs validation accuracy with and without dropout on a dataset small enough to overfit easily.
4. Implement a learning rate scheduler that halves the learning rate every 5 epochs, and plot the effective learning rate over 20 epochs alongside the training loss.

## Key Terms

| Term | What it actually means |
|---|---|
| Cross-entropy loss | The standard classification loss, penalizing confident wrong predictions heavily; pairs naturally with softmax outputs |
| Epoch | One complete pass through the entire training dataset |
| Adam | An optimizer combining momentum with a per-parameter adaptive learning rate; a common default for training CNNs from scratch |
| Weight decay | An L2 penalty added to the loss to discourage large weights, reducing overfitting |
| Dropout | Randomly zeroing a fraction of units during training to prevent over-reliance on any single unit |
| Batch normalization | Normalizing a layer's activations across the current batch, stabilizing and speeding up training |
| Learning rate schedule | A rule for changing the learning rate over the course of training, typically decreasing it as training progresses |
