# 38. Causal Representation Learning

## Learning Objectives

- Explain the gap causal representation learning tries to close: causal models usually assume the variables are already given
- State the identifiability problem for latent variable models and why unsupervised ICA/VAEs are not identifiable in general
- Describe how auxiliary information (interventions, time, or side labels) restores identifiability (iVAE / nonlinear ICA)
- Distinguish disentanglement metrics from causal identifiability

---

## 1. The Problem

Every earlier module assumed $T$, $Y$, and $X$ are well-defined variables with known causal roles. In images, text, or sensor streams, the *variables* are not given — a photo is a grid of pixels, not "object identity" and "lighting". **Causal representation learning** asks how to learn a small set of variables from raw data such that causal reasoning (do-operators, DAGs) can be applied to them.

## 2. The Concept

### 2.1 Latent generative model

$$Z\sim P(Z),\qquad X=g(Z),$$
with $g$ an unknown (often nonlinear) mixing function producing the observed data $X$ from latent causal factors $Z$ (e.g., object shape, position, lighting). The goal is to recover $Z$ (up to relabeling/scaling) from $X$ alone or with weak supervision.

### 2.2 Non-identifiability of nonlinear ICA

Hyvärinen and Pajunen (1999) showed that for i.i.d. data with an arbitrary nonlinear $g$, **any** distribution can be produced by *infinitely many* different $(P(Z),g)$ pairs, including ones where the recovered "latents" are meaningless rotations of each other. Without further structure, unsupervised deep generative models (vanilla VAEs, GANs) cannot be guaranteed to recover the true causal factors — a good reconstruction loss does not imply a causally meaningful representation.

### 2.3 Restoring identifiability with auxiliary information

Identifiable variants condition the latent prior on an observed auxiliary variable $U$ (time index, environment label, or intervention indicator), $P(Z\mid U)$, assumed to have an exponential-family form. Khemakhem et al. (2020, iVAE) show that if $Z_i\perp Z_j\mid U$ and $U$ takes enough distinct values, $Z$ is identifiable up to a permutation and componentwise transform. **Interventions or environment labels are exactly the auxiliary variable that makes recovery possible** — the same insight as multi-environment invariance methods (invariant risk minimization) used for out-of-distribution robustness.

### 2.4 Disentanglement vs. identifiability

A metric like the Mutual Information Gap can be high on a representation that is well "disentangled" empirically yet still not equal (even up to relabeling) to the *true* causal factors, because different random seeds of the same unsupervised model recover different, equally-plausible-looking latents (Locatello et al., 2019). Identifiability is a mathematical guarantee about recoverability; disentanglement scores are empirical descriptions of one particular learned representation.

## 3. Use It: Code

`code/identifiability_demo.py` uses the cleanest classical illustration of §2.2's point: two **Gaussian** independent latent sources (Gaussian sources are exactly the case where ordinary ICA/PCA cannot identify a unique rotation — any orthogonal rotation of an isotropic-in-the-wrong-basis Gaussian looks the same), linearly mixed into 2D observations. It shows (a) plain PCA on the pooled data recovers each true source with only moderate correlation (the rotation really is ambiguous), while (b) using the auxiliary environment label — here, two environments where the sources have *different* variances — and jointly diagonalizing the two environments' covariance matrices recovers each true source's direction almost perfectly. This is the linear special case of the same principle iVAE uses in the nonlinear, non-Gaussian setting from §2.3: an auxiliary variable that changes the latent distribution across settings is what breaks the identifiability logjam.

## Exercises

1. Set both environments to the *same* variances. Does joint diagonalization still recover the sources? Why or why not, given §2.3's requirement that $U$ take "enough distinct values" in a meaningful sense?
2. Add a third environment with yet another variance pattern, and extend the joint-diagonalization approach to use pairs of environments. Does using more environments make the recovery more robust to noise?
3. In your own words, explain why "reconstructs the input well" (a property any of PCA's rotations would have, since they span the same subspace) is not evidence that a representation is causally meaningful.

## Key Terms

| Term | What it actually means |
|---|---|
| Causal representation learning | Learning latent variables from raw data such that causal structure (a DAG, interventions) can be defined on them |
| Non-identifiability (nonlinear ICA) | Many different latent variable models fit the same observed distribution equally well |
| Auxiliary variable | Side information (environment, time, intervention label) that restores identifiability |
