# 16. RNN Regularization

## Learning Objectives

- Explain why naive dropout breaks an RNN's recurrent connections, and how variational dropout fixes this
- Apply weight decay, gradient clipping, and early stopping specifically in the context of RNN training
- Combine these techniques into a complete, overfitting-resistant RNN training setup

## The Problem

RNNs — especially stacked (Lesson 10) or large-hidden-size ones — have plenty of capacity to overfit, exactly like the CNN module's networks. Most of the general regularization playbook (dropout, weight decay, early stopping — all covered generally in the CNN module's Lesson 10) applies to RNNs too, but naively applying dropout to a recurrent connection specifically causes a problem unique to sequential architectures: it can directly interfere with the mechanism that makes an RNN work at all.

## The Concept

### Why naive dropout on recurrent connections is a problem

Standard dropout (CNN module, Lesson 10) randomly zeroes a *different* random subset of units on every forward pass. Applied straightforwardly to an RNN's hidden-state-to-hidden-state connection (`W_hh @ h_(t-1)` in Lesson 03's cell equation), this means a *different* random mask is applied at *every single time step* — so information the network is trying to carry forward from step 3 to step 4 might survive being dropped at step 4, only to be randomly dropped again at step 5, and so on. Over many time steps, this repeated, uncorrelated random dropping systematically destroys the network's ability to maintain any long-term signal at all — directly undermining the entire point of the recurrent connection, and actively working against the same "preserve information across steps" goal that motivated LSTM and GRU (Lessons 07-08) in the first place.

```
Naive dropout on the recurrent connection, per time step:

t=1: mask_1 (random)  -> h_1 partially zeroed
t=2: mask_2 (DIFFERENT random mask)  -> h_2 partially zeroed, but different units than t=1
t=3: mask_3 (DIFFERENT again)  -> ...

Result: no single unit reliably survives across many steps -- long-term
memory is disrupted by the very technique meant to reduce overfitting.
```

This is exactly why Lesson 10 specifically noted dropout is standard *between* stacked layers but not naively within a single layer's own recurrence — this lesson explains precisely why, and gives the fix for cases where recurrent dropout genuinely is wanted.

### Variational dropout: the same mask at every time step

The fix (Gal & Ghahramani, 2016): use the *same* dropout mask across every time step within one forward pass (though still a different, freshly-random mask on each new training example/batch), rather than resampling it at every step.

```
Variational dropout on the recurrent connection, per time step:

mask (sampled ONCE per training example/batch, reused at every step)

t=1: mask -> h_1 partially zeroed by mask
t=2: mask (SAME mask) -> h_2 zeroed at the SAME positions
t=3: mask (SAME mask) -> h_3 zeroed at the SAME positions
...

Result: a unit that's "kept" is kept consistently across the whole
sequence for this example, so long-term signals CAN still propagate
through it -- only the recurrent CONNECTION strength is regularized,
not the network's ability to remember anything at all.
```

Because the same units are consistently zeroed (or consistently kept) throughout the sequence, the network can still learn to route important long-term information through the units that happen to survive for this particular forward pass, rather than having that routing disrupted at every single step. `nn.LSTM`'s built-in `dropout` argument, as covered in Lesson 10, actually applies to the connections *between* stacked layers, not to the recurrent connection within a layer — implementing true variational recurrent dropout requires either a custom cell implementation or a library specifically supporting it, which the code demo below implements directly to make the distinction concrete.

### Weight decay

Weight decay (L2 regularization, covered generally in the CNN module's Lesson 10) applies to RNNs exactly as it does to any other network — penalizing the squared magnitude of weights to discourage overly large values:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

No RNN-specific nuance here; it's applied the same way as any other architecture, though in practice it's often used more conservatively for RNNs than for CNNs, since keeping recurrent weights small also interacts with the gradient-flow considerations from Lesson 06 — a heavily-decayed `W_hh` could also make vanishing gradients slightly more likely, so this is one more parameter worth tuning rather than defaulting blindly.

### Gradient clipping as regularization-adjacent practice

Gradient clipping (Lesson 06) isn't a regularizer in the traditional "reduce overfitting" sense, but it's worth including here since it's just as standard a piece of a robust RNN training setup as dropout and weight decay — without it, occasional large gradient spikes (more common in RNNs than in typical feedforward or convolutional networks, given the repeated-multiplication dynamics from Lesson 06) can effectively undo many steps of otherwise-careful, well-regularized training in one bad update.

### Early stopping

Also general practice (not RNN-specific), but especially relevant given how sensitive some RNN training runs can be to instability: track validation loss during training and stop (or restore the best-seen checkpoint) once it stops improving, rather than training for a fixed, possibly-too-long number of epochs regardless of what validation performance is actually doing.

```python
best_val_loss = float("inf")
patience_counter = 0
patience = 5

for epoch in range(max_epochs):
    train_one_epoch(...)
    val_loss = evaluate(...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break   # stop training, restore the best checkpoint
```

### Putting it together

A complete, reasonably overfitting-resistant RNN training setup combines: inter-layer dropout for a stacked architecture (Lesson 10), variational dropout on the recurrent connection if regularizing within-layer memory specifically is needed, modest weight decay, gradient clipping (standard regardless of overfitting concerns, per Lesson 06), and early stopping based on validation loss — none of these individually solves overfitting on its own, but combined they form the same kind of layered defense the CNN module's Lesson 10 built up for convolutional networks, adapted for the specific ways recurrence can go wrong.

See `code/rnn_regularization_demo.py` for a from-scratch variational dropout RNN cell demonstrating the mask-reuse mechanism directly, a comparison against naive per-step dropout showing the long-term-signal disruption concretely, and a complete training loop combining weight decay, gradient clipping, and early stopping.

## Exercises

1. Implement a variational dropout RNN cell (same mask reused across all time steps in one forward pass) and confirm the mask differs between two separate forward passes (two different training examples) but stays fixed within one.
2. Construct a synthetic long-range-dependency task (similar to Lesson 06's exercises) and compare a model using naive per-step recurrent dropout against one using variational dropout, at a fixed dropout rate. Confirm variational dropout preserves more of the network's ability to solve the task.
3. Train an RNN with and without weight decay on a dataset small enough to overfit, and compare the gap between training and validation loss in each case.
4. Implement early stopping with a patience parameter and confirm it stops training before the maximum epoch count on a run where validation loss starts increasing partway through training.

## Key Terms

| Term | What it actually means |
|---|---|
| Naive dropout (on a recurrent connection) | Applying a fresh, independently-random dropout mask at every time step, which disrupts an RNN's ability to carry information across steps |
| Variational dropout | Reusing the same dropout mask across every time step within one forward pass, preserving the ability to carry long-term signals through the units that survive |
| Weight decay | An L2 penalty on weight magnitude, applied the same way to RNNs as to any other architecture |
| Early stopping | Halting training (or restoring an earlier checkpoint) once validation performance stops improving, rather than training for a fixed number of epochs regardless |
| Patience | The number of epochs without validation improvement tolerated before early stopping triggers |
