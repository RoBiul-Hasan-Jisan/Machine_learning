# 05 · Model-Agnostic Global Methods: PDP & ICE

SHAP and LIME explain individual predictions. **PDP** and **ICE** zoom out to show how a feature affects predictions *across the whole dataset* — pure global, model-agnostic explanations that need no special library, just `sklearn.inspection`.

## What you'll learn
- **Partial Dependence Plots (PDP):** average predicted outcome as one feature varies, holding others at their marginal distribution — "on average, how does this feature drive the prediction?"
- **Individual Conditional Expectation (ICE):** the same idea but *per instance* rather than averaged — reveals whether the average PDP curve is hiding heterogeneous subgroups (e.g. the feature helps some patients and hurts others, averaging to "no effect")

## Run it
[`05-model-agnostic-pdp-ice.ipynb`](../../code/05-model-agnostic-pdp-ice.ipynb)

## Key takeaway
PDP can be misleading when there's feature interaction or heterogeneity — always look at the ICE lines underneath the PDP average before trusting it. If ICE lines fan out wildly or cross, the PDP's average trend does not represent most individuals.

## Next
[`06-counterfactual-explanations`](../06-counterfactual-explanations) — instead of "how does the feature affect predictions in general," ask "what's the smallest change needed to flip *this* prediction?"
