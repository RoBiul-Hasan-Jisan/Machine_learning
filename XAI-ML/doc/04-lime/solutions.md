# Solutions — 04 LIME

1. No — because LIME samples random perturbations around the instance each time, the exact weights (and sometimes even the top feature ranking) can shift slightly between runs. This instability is one of LIME's well-documented weaknesses, in contrast to SHAP/TreeSHAP, which is deterministic for a given model and instance.

2. The default `kernel_width` is typically `sqrt(num_features) * 0.75`. A much smaller kernel width makes the "neighborhood" tighter (only very close perturbed samples get high weight), producing a more locally faithful but potentially noisier explanation. A much larger kernel width blends in farther, less-similar samples, producing a smoother but less locally accurate explanation — it starts to resemble a more global linear approximation.

3. The top 3 features usually stay fairly consistent since they carry the most weight in the local linear model, but exact numeric weights can shift because the underlying ridge regression refits with a different number of active features. Very small `num_features` can hide interactions the larger sets reveal.

4. They generally agree on the broad top features (e.g. `worst concave points`, `worst area`) since both are approximating the same model's local behavior, but exact rankings and magnitudes typically differ — SHAP obeys strict additivity/consistency axioms while LIME's coefficients come from a locally-fit linear surrogate on randomly sampled perturbations, so it can weight things differently, especially when features are correlated.

5. Example: a real-time fraud-scoring API that must return an explanation alongside every transaction decision within tens of milliseconds, at high request volume. TreeSHAP is fast for tree models but SHAP in general (especially `KernelExplainer` for non-tree models) can be far slower than LIME, which needs only a handful of perturbed samples and a cheap linear fit — making LIME preferable when speed and model-agnosticism matter more than exact theoretical guarantees.
