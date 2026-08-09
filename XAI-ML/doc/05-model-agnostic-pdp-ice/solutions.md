# Solutions — 05 PDP & ICE

1. Typically the PDP for `mean area` shows predicted probability of Benign decreasing as `mean area` increases — larger tumor area is associated with a higher chance of malignancy in this model, consistent with clinical intuition.

2. Usually most ICE lines follow the same general downward trend as the PDP average, but some lines will be flatter or steeper, and a few may plateau earlier or later — this heterogeneity shows the effect of `mean area` isn't perfectly uniform across all patients; it likely interacts with other features (e.g. a patient with otherwise very "benign-looking" values may need a much larger area increase before their prediction flips).

3. `worst radius`'s PDP will typically look similar in shape (also a downward-sloping curve) since tumor "worst" radius and mean area are both proxies for tumor size and are highly correlated — the model has likely learned to rely on tumor size broadly, so any size-related feature shows a similar pattern.

4. A 2D PDP contour plot would show whether the *combination* of high `mean area` and high `mean smoothness` predicts differently than either alone (e.g. an interaction effect: a large but smooth tumor might be predicted very differently than a large and rough one) — something invisible in two separate 1D PDP plots, which only show marginal (averaged) effects.

5. Because PDP computes its curve by fixing the feature of interest and averaging the prediction over the *empirical* distribution of all other features (holding them at their real observed values while varying just the one feature) — but when varying `mean area` while holding `mean radius`/`mean perimeter` fixed at real values, it creates unrealistic synthetic combinations (e.g. a huge area with a tiny radius), since these features are geometrically dependent. This means PDP for correlated features can be misleading, evaluating the model on out-of-distribution combinations it never saw in training.
