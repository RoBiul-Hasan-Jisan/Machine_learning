# 40. Causal VLM / Multimodal Causal Inference

## Learning Objectives

- Explain the two main ways causal questions arise with vision-language and multimodal models: as *subjects of study* and as *tools for confound control*
- Describe why text- or image-derived variables are typically **proxies** for the causal factors of interest, and what that costs
- Use a foundation model's embeddings as a (partial) confounder-adjustment covariate, and see why "partial" is not "complete"
- Identify spurious shortcuts in multimodal training data, connecting back to Module 39's invariance ideas

---

## 1. The Problem

Vision-language models (VLMs) and other multimodal systems raise causal questions on two distinct fronts: (i) **causal inference using unstructured data**, where images or text stand in for otherwise-unmeasured confounders or outcomes (e.g., using satellite images as a proxy for economic development, or clinical notes as a proxy for disease severity); and (ii) **causal inference about the models themselves**, asking whether a VLM's behavior reflects a genuine causal understanding of a scene versus a correlational shortcut (e.g., recognizing "cow" from grass backgrounds rather than the animal). Both connect directly to Module 38 (recovering causal variables from raw data) and Module 39 (spurious shortcuts and invariance).

## 2. The Concept

### 2.1 Unstructured data as a proxy variable

Suppose the true confounder is $Z$ (e.g., "neighborhood socioeconomic status"), unmeasured directly, but an image $I$ or a text description is available and $Z$ influences $I$. A model $\hat Z=f(I)$ trained to predict some correlate of $Z$ gives a **proxy**, not $Z$ itself. If $\hat Z$ captures only part of $Z$'s variation (measurement error) or is influenced by something else entirely (e.g., camera/lighting artifacts unrelated to $Z$), adjusting for $\hat Z$ only partially blocks the backdoor path $T\leftarrow Z\to Y$ — classical results on covariate measurement error (non-differential error attenuates bias correction, generally *toward* the naive unadjusted estimate rather than fully removing it) apply directly.

### 2.2 Partial deconfounding: better than nothing, not as good as $Z$

If $\hat Z=Z+\text{noise}$ (classical measurement error, uncorrelated with everything else), adjusting for $\hat Z$ removes *some* but not all of the confounding bias, with the remaining bias shrinking as $\hat Z$'s correlation with the true $Z$ increases. This gives a concrete, checkable recommendation: report how strongly $\hat Z$ predicts other known proxies or outcomes plausibly driven by $Z$, as indirect evidence for how much of $Z$'s variation it actually captures — while being explicit that full deconfounding is not guaranteed.

### 2.3 Shortcut learning in multimodal training data

Multimodal training data (image–caption pairs scraped from the web, for instance) reflects the correlational structure of *how images and text co-occur online*, not a curated causal experiment. A model can learn that a caption word co-occurs with an incidental visual feature (a watermark, a background object, a photographer's stylistic choice) rather than the depicted concept itself — precisely a spurious correlation in Module 39's sense, except now the "environment" is implicit in different data sources, platforms, or time periods, which is rarely labeled explicitly the way Module 39's synthetic environments were.

### 2.4 Using embeddings for confounder control, carefully

A common practical pattern: use a pretrained model's embedding of an image or document as a covariate in an adjustment set, hoping it captures unmeasured visual or textual confounders. This can help (§2.2), but the backdoor criterion (Module 14) still requires the embedding to actually block every backdoor path — an embedding trained for a different task (e.g., object recognition) may discard exactly the visual information that mattered for confounding while retaining information that is causally irrelevant. Reporting *what the embedding predicts well and poorly* is more informative than asserting "confounders were controlled for using image embeddings."

## 3. Use It: Code

`code/proxy_confounder_demo.py` simulates a true confounder $Z$, an outcome and treatment depending on it, and a noisy "embedding-style" proxy $\hat Z$ with tunable measurement-error variance, then shows the bias-reduction curve as the proxy's quality improves from pure noise to a near-perfect copy of $Z$ — the concrete version of §2.2's "shrinks but does not vanish" claim.

## Exercises

1. At what proxy-quality level (noise variance) does adjusting for $\hat Z$ remove roughly half of the naive bias? Does this match a simple attenuation-factor calculation?
2. Suppose two different embedding models give proxies with different noise levels for the same $Z$. How would you decide which one to trust more, using only observable checks (no access to true $Z$)?
3. Connect this module's §2.3 back to Module 39 §2.1: in what sense is "the year a photo was taken" or "the platform it was scraped from" acting as an unlabeled environment variable?

## Key Terms

| Term | What it actually means |
|---|---|
| Proxy variable | An observed variable (often derived from unstructured data) that imperfectly stands in for an unmeasured causal variable |
| Non-differential measurement error | Error in a proxy that is unrelated to treatment and outcome given the true variable; typically attenuates rather than eliminates confounding bias |
| Shortcut learning | A model relying on a correlate of the concept of interest rather than the concept itself |
