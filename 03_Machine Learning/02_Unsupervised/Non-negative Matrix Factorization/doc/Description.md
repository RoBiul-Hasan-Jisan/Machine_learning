# Non-negative Matrix Factorization (NMF)

## 1. Intuition
NMF factorizes a **non-negative** data matrix `X` (n samples × d features,
all entries ≥ 0) into two smaller non-negative matrices:

```
X ≈ W @ H       where W is (n × k), H is (k × d), and W, H >= 0 everywhere
```

- Each row of `H` is a "part"/"topic"/"basis component" (in the original
  d-dimensional feature space).
- Each row of `W` gives the non-negative *weights/coefficients* of how much
  of each part is present in that sample.

Because everything is non-negative, samples are reconstructed as a
**purely additive** combination of parts — no subtraction allowed. This
matches how many real-world signals actually work (pixel intensities,
word counts, spectral data are all naturally non-negative), and tends to
produce much more **interpretable, "parts-based" components** than PCA,
whose components can have negative entries and are usually only
interpretable in combination with each other.

## 2. Classic example: face images (Lee & Seung, 1999)
Applying NMF to a dataset of face images (pixels ≥ 0) tends to discover
components that look like localized face **parts** — eyes, nose, mouth
regions — because faces really are built by combining these parts
additively. PCA on the same data instead produces global, ghostly
"eigenfaces" with positive and negative pixel values that are harder to
interpret individually.

## 3. The optimization problem
Minimize the reconstruction error subject to `W, H >= 0`:

```
min_{W,H >= 0}  || X - W H ||_F^2      (Frobenius norm / squared error)
```

(An alternative common objective is the **KL-divergence** /
generalized-Poisson loss instead of squared error — better suited when `X`
represents counts, e.g. word counts in documents.)

This is a **non-convex** problem (jointly in W and H) — unlike PCA, there's
no closed-form solution, and different initializations can converge to
different local optima.

## 4. Algorithm — Multiplicative Update Rules (Lee & Seung)
The most common, simple-to-implement approach. Starting from random
non-negative `W`, `H`, alternate:

```
H <- H * (W^T X) / (W^T W H + eps)
W <- W * (X H^T) / (W H H^T + eps)
```

(`eps` is a small constant to avoid division by zero.) Each update is
guaranteed to never increase the reconstruction error, and multiplication
by a non-negative factor keeps `W, H` non-negative automatically — this is
why it's such an elegant algorithm.

## 5. Complexity
- Each iteration: dominated by matrix multiplications, roughly
  `O(n*d*k)` per update — comparable to a K-Means iteration; run for many
  iterations until convergence (change in reconstruction error is small).

## 6. Strengths / Weaknesses
**Strengths**:
- Interpretable, additive, "parts-based" components — great for topic
  modeling (text), image parts, audio spectrogram decomposition (source
  separation).
- Naturally suited to sparse, non-negative data (word counts, pixel
  intensities, gene expression levels).

**Weaknesses**:
- Only applicable to non-negative data (or requires shifting into that
  form, which can be awkward).
- Non-convex — sensitive to initialization, no unique/global solution
  (unlike PCA's orthogonal, ordered components).
- No orthogonality guarantee between components (they can overlap /
  correlate).

## 7. Relation to other methods
- Like PCA, it's a **linear** factorization/dimensionality-reduction
  method — but with a different constraint (non-negativity vs.
  orthogonality + variance-maximization).
- Widely used for **topic modeling** in NLP (each topic = a non-negative
  distribution over words; each document = a non-negative mix of topics) —
  a simpler alternative to Latent Dirichlet Allocation (LDA).

## 8. Basic → Pro
1. **Basic**: multiplicative update rules, squared-error objective, image
   parts visualization.
2. **Intermediate**: KL-divergence objective (better for count data like
   word frequencies); multiple random restarts to deal with non-convexity.
3. **Pro**: NMF for **topic modeling** on real text (TF-IDF or count
   matrix as `X`, inspect top words per topic from `H`); NMF for **audio
   source separation** (magnitude spectrogram as `X`).

## 9. Code in this folder
- `code/nmf_scratch.py` — NMF from scratch in NumPy using multiplicative
  update rules, with reconstruction-error tracking.
- `code/nmf_sklearn_demo.py` — `sklearn.decomposition.NMF` for topic
  modeling on a small text corpus (prints top words per discovered topic)
  and for face-image parts decomposition compared against PCA.
