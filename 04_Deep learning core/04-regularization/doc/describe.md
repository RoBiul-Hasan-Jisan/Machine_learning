# 04 · Regularization

A well-optimized network can still fail — by memorizing the training set instead of learning generalizable patterns. Regularization is the set of techniques that trade a little training accuracy for a lot more test-time (real-world) accuracy.

## What you'll learn
- **L1 Regularization** — penalizing the sum of absolute weight values
- **L2 Regularization** — penalizing the sum of squared weight values
- **Weight Decay** — L2 regularization's optimizer-side equivalent (and why it isn't *always* identical, see lesson 03's AdamW)
- **Dropout** — randomly disabling neurons during training
- **Early Stopping** — simply stopping training before you overfit
- **Data Augmentation** — manufacturing more (varied) training data
- **Batch Normalization** — normalizing activations between layers

---

## 1. L1 Regularization

Adds the sum of absolute weight values to the loss:

```
L_total = L_original + λ·Σ|wᵢ|
```

**Effect:** encourages **sparsity** — many weights get pushed exactly to zero, effectively performing automatic feature selection. Useful when you suspect many input features are irrelevant.

## 2. L2 Regularization

Adds the sum of *squared* weight values to the loss:

```
L_total = L_original + λ·Σwᵢ²
```

**Effect:** shrinks all weights toward zero smoothly, but rarely to exactly zero — it discourages any single weight from becoming too large (which tends to correspond to over-reliance on a specific feature/pattern), spreading influence more evenly. This is the more common choice in deep learning.

**L1 vs L2 intuition:** L1's gradient is constant (`±λ`) regardless of weight size, so it keeps pushing small weights all the way to zero. L2's gradient (`2λw`) shrinks as the weight shrinks, so it asymptotically approaches — but rarely reaches — zero.

## 3. Weight Decay

Weight decay directly shrinks weights toward zero at every optimizer step, independent of the loss gradient:

```
W := W - η·λ·W    (the decay term)
```

For plain SGD, adding `λw` to the gradient (L2 regularization) is *mathematically equivalent* to weight decay. This equivalence famously breaks for Adam (see lesson 03's AdamW) — which is exactly why AdamW's *decoupled* weight decay was introduced.

## 4. Dropout

During training, **randomly zero out** a fraction `p` of neurons in a layer on every forward pass (a different random subset each time):

```
during training:  a := a * mask,   mask ~ Bernoulli(1-p), scaled by 1/(1-p)
during inference: use all neurons, no masking (scaling above keeps expected activation magnitude consistent)
```

**Why it works:** it prevents neurons from co-adapting too heavily on specific other neurons (relying on a "team" of neurons that happen to work together on the training set). Effectively, dropout trains an exponential ensemble of thinned sub-networks simultaneously, and at inference time you get an approximation of averaging all of them.

## 5. Early Stopping

Monitor validation loss during training; stop (and keep the best checkpoint) once it stops improving for a set number of epochs (the "patience"), even if training loss is still decreasing.

**Why it works:** training loss almost always keeps decreasing the longer you train (the model keeps fitting the training data more closely), but validation loss typically decreases, bottoms out, then starts rising again as the model begins memorizing training-set-specific noise instead of general patterns. Early stopping catches the model at that turning point.

## 6. Data Augmentation

Artificially expand the effective size and diversity of the training set by applying label-preserving transformations: random crops/flips/rotations/color jitter for images, synonym replacement/back-translation for text, time-warping/noise-injection for time series, etc.

**Why it works:** overfitting happens partly because the model sees too few *variations* of each underlying pattern. Augmentation exposes the model to many more variations without needing to collect more real data, making it harder to simply memorize any single exact training example.

## 7. Batch Normalization

Normalizes each layer's activations (zero mean, unit variance) across a mini-batch, then applies a learnable scale `γ` and shift `β`:

```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γ·x̂ + β                          (γ, β are learned parameters)
```

**Why it's (partly) a regularizer:** normalization statistics (`μ_batch`, `σ²_batch`) are computed from a *randomly sampled mini-batch*, so they're slightly noisy from batch to batch — this injects a small amount of noise into training, similar in spirit to dropout, which has a mild regularizing effect. (Batch norm's primary purpose is actually training stability/speed — see the "why" note below — but its regularizing side-effect is well documented.)

**The primary reason it was introduced:** to combat "internal covariate shift" — as earlier layers' weights update, the *distribution* of inputs to later layers keeps shifting, forcing those later layers to constantly re-adapt. Normalizing activations between layers keeps their distribution more stable throughout training, which in practice allows much higher learning rates and faster convergence, on top of the regularization side-benefit.

## Run the code
[`04-regularization.ipynb`] — implements L1/L2 penalties, Dropout, and Batch Normalization from scratch in NumPy, demonstrates early stopping and data augmentation, then reproduces the comparison with PyTorch's `nn.Dropout`, `nn.BatchNorm1d`, and `weight_decay`.

