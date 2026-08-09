# 09 · Building a Complete XAI Dashboard + Trust Metrics

A capstone lesson: combine everything from `03`–`06` into one reusable class, then quantify how much you should actually *trust* the model.

## What you'll learn
- How to build an `XAIDashboard` class that wraps SHAP + LIME + what-if analysis behind a single, reusable interface for any scikit-learn-compatible model
- How to compute a composite **trust score** from accuracy, local consistency (prediction stability under small noise), and confidence calibration (is the model more confident when it's right than when it's wrong?)

## Run it
[`09-xai-dashboard-and-trust.ipynb`](../../code/09-xai-dashboard-and-trust.ipynb)

## Key takeaway
"Explainable" and "trustworthy" are not the same thing. A model can produce clean, plausible-looking SHAP/LIME explanations while still being poorly calibrated or unstable. Trust metrics like the ones here are a sanity check *on top of* explanations, not a replacement for them.

## Next
[`10-modalities-regulation-ethics`](../10-modalities-regulation-ethics) and [`11-limitations-and-open-problems`](../11-limitations-and-open-problems) for the bigger picture: other data types, the law, and where XAI still falls short.
