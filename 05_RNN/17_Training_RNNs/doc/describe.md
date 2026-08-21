# 17. Training RNNs

## Learning Objectives

- Assemble forward propagation, loss computation, BPTT, and an optimizer into a complete RNN training loop
- Apply the RNN-specific practical techniques this module has built up: gradient clipping, dropout placement, truncated BPTT
- Diagnose common RNN training failures using symptoms specific to sequential models

## The Problem

Lessons 03-16 covered the pieces: the cell mechanics, BPTT, the gradient problems and their architectural fixes (LSTM/GRU), bidirectionality, depth, embeddings, and regularization. Training an RNN in practice means combining all of these correctly, plus knowing which symptom points to which fix — this lesson is the practical synthesis, mirroring the CNN module's Lesson 10 (Training CNNs) but for the sequential-model-specific issues this module has covered.

## The Concept

### The training loop

```python
for epoch in range(num_epochs):
    for X_batch, y_batch, lengths in train_loader:      # Lesson 02's padded batches
        optimizer.zero_grad()
        output, hidden = model(X_batch)                  # forward pass (Lessons 03-04)
        loss = loss_fn(output, y_batch)                  # e.g. cross-entropy, masked per Lesson 02
        loss.backward()                                   # BPTT (Lesson 05), autograd handles it
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Lesson 06
        optimizer.step()
```

Every line traces back to a specific earlier lesson. The one RNN-specific addition relative to the CNN module's training loop is gradient clipping (Lesson 06) — standard practice for RNNs in a way it typically isn't for CNNs, precisely because of BPTT's repeated-multiplication structure.

### Masking the loss for padded sequences

Lesson 02 introduced padding and masks for handling variable-length sequences in a batch. The loss computation needs to actually *use* that mask — otherwise the model is penalized (or rewarded) for its predictions on meaningless padding positions, which both distorts the training signal and can teach the model to output whatever minimizes loss on padding (usually irrelevant, sometimes actively unhelpful) rather than focusing entirely on real positions:

```python
def masked_loss(logits, targets, mask, loss_fn):
    losses = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))  # per-position loss, unreduced
    losses = losses * mask.view(-1)                                        # zero out padded positions
    return losses.sum() / mask.sum()                                       # average over REAL positions only
```

This requires calling the loss function with `reduction="none"` so it returns a per-position loss rather than an already-averaged scalar, then applying the mask and averaging manually — a detail easy to miss, and a common source of subtly wrong training runs where loss curves look reasonable but the model underperforms because padding positions were quietly polluting the gradient.

### Choosing sequence length and truncated BPTT in practice

For long sequences, Lesson 05's truncated BPTT is a real, standard choice, not just a theoretical option: split long sequences (or long streams, like a book being trained on continuously) into fixed-length chunks, and carry the hidden state — but explicitly detach it from the computation graph — across chunk boundaries:

```python
hidden = model.init_hidden(batch_size)
for chunk in chunks:
    hidden = hidden.detach()          # cut the gradient history from the previous chunk
    optimizer.zero_grad()
    output, hidden = model(chunk, hidden)
    loss = loss_fn(output, chunk_targets)
    loss.backward()
    optimizer.step()
```

`.detach()` is the specific mechanism: it keeps the hidden state's *values* (so the model still has continuity of memory across chunks) while preventing gradients from flowing back through it into the previous chunk's computation — exactly the truncated-BPTT tradeoff from Lesson 05, made concrete in code.

### Diagnosing RNN-specific training failures

Building on the CNN module's general troubleshooting checklist (its Lesson 19), several failure modes are specific to sequential models:

| Symptom | Likely cause | Where to look |
|---|---|---|
| Loss becomes NaN, especially on longer sequences | Exploding gradients | Lesson 06 (gradient clipping) |
| Model performs well on short sequences, poorly on long ones | Vanishing gradients, plain RNN can't capture long-range dependencies | Lesson 07/08 (switch to LSTM/GRU) |
| Loss looks fine but predictions are nonsensical near sequence ends | Padded positions leaking into the loss | This lesson's masked loss section; re-check the mask is actually applied |
| Bidirectional model works in training/validation but fails badly in deployment | Bidirectionality used on a task that's actually online/streaming | Lesson 09 (bidirectionality requires the full sequence up front) |
| Training loss decreases, validation loss increases quickly | Overfitting, common with stacked/high-capacity RNNs on limited data | Lesson 16 (variational dropout, weight decay, early stopping) |
| Training is extremely slow, especially on long sequences | BPTT's cost scaling with sequence length | Lesson 05 (truncated BPTT), or reduce sequence length (Lesson 02's truncation) |
| Model repeats the same token/value endlessly during generation | Greedy decoding collapsing into a loop | Lesson 14 (sampling strategies: temperature, top-k) |

The single most valuable habit, echoing the CNN module's closing advice: inspect actual model outputs on a handful of validation examples throughout training, not just the aggregate loss curve. A dropping loss curve can hide a model that's learned to exploit padding, or a bidirectional model that would fail immediately in the online setting it's actually meant for — problems that are obvious from a few real examples and easy to miss from a single scalar metric.

### A complete, minimal training script

See `code/train_rnn_demo.py` for an end-to-end training loop: masked loss on padded sequences, gradient clipping, truncated BPTT with `.detach()`, and a validation loop, applied to a small synthetic sequence classification task — combining every piece introduced across this section of the module.

## Exercises

1. Implement `masked_loss` and confirm, on a batch with heavy padding (e.g. most sequences much shorter than the batch's max length), that the masked and unmasked loss values differ meaningfully.
2. Implement truncated BPTT with `.detach()` on a long synthetic sequence, and confirm (by checking `.grad_fn` or attempting to call `.backward()` twice) that gradients genuinely do not flow across the chunk boundary.
3. Deliberately omit gradient clipping on a task with W_hh scaled up (Lesson 06's exploding-gradient setup) and confirm training produces `NaN` loss; then add clipping back and confirm training stabilizes.
4. Using the diagnostic table above, write a short "incident report" for a hypothetical RNN project where validation accuracy is high but a bidirectional model was deployed for real-time transcription and failed. Identify the root cause and the fix.

## Key Terms

| Term | What it actually means |
|---|---|
| Masked loss | A loss computation that excludes padded positions from the average, computed by zeroing per-position losses at padded positions before averaging over only real positions |
| detach() | The PyTorch operation that keeps a tensor's values while cutting it off from the autograd computation graph, used to implement truncated BPTT across chunk boundaries |
| Training diagnostic | A specific observable symptom during training used to identify which underlying architectural or procedural issue is responsible |
