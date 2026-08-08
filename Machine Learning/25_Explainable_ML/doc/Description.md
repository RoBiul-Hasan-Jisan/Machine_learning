# Explainable ML: 

How to understand *why* a model makes the predictions it does — from simple built-in importances through SHAP, LIME, and dependence plots.

---

## 0. Why Explainability Matters

- **Trust & debugging** — catch leakage, spurious correlations, or bugs before deployment.
- **Regulatory/compliance** — many domains (finance, healthcare, hiring) legally require explanations for automated decisions.
- **Stakeholder communication** — domain experts and business users need to understand model behavior, not just accuracy numbers.
- **Model improvement** — understanding *what* the model relies on often reveals what feature engineering to try next.

**Two key distinctions:**
| | Global | Local |
|---|---|---|
| **Meaning** | How the model behaves overall, across the whole dataset | Why the model made *this specific* prediction for *this specific* row |
| **Examples** | Feature importance, permutation importance, PDP | SHAP (per-instance), LIME, ICE |

| | Model-specific | Model-agnostic |
|---|---|---|
| **Meaning** | Only works for certain model types | Works on any black-box model |
| **Examples** | Tree-based feature importance, linear coefficients | Permutation importance, SHAP (mostly), LIME, PDP, ICE |

---

## 1. Feature Importance (Built-in / Model-Specific)

### Tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM)
Importance is typically computed from how much each feature reduces impurity (Gini/entropy) or loss, summed across all splits that use it, weighted by how many samples pass through that split.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).plot.bar()
```

**Known weaknesses:**
- **Biased toward high-cardinality features** — features with more unique values get more chances to be selected for splits, inflating their apparent importance even if not truly predictive.
- **Biased by correlated features** — importance gets split/diluted among correlated features, understating their true collective effect.
- Computed *only* from training data — can reflect overfitting rather than genuine predictive value.

### Linear models
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train_scaled, y_train)  # must scale features first!
coef_importance = pd.Series(model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
```
> Coefficients are only comparable across features if the features were **standardized first** — otherwise a feature's raw scale (e.g., income in dollars vs. age in years) distorts the coefficient magnitude.

**Coefficient sign and magnitude:** positive coefficient → increases the log-odds of the positive class; magnitude reflects strength *given the other features are held constant* — but says nothing about interaction effects or non-linearities.

---

## 2. Permutation Importance

A **model-agnostic** method: works for any fitted model, not just trees/linear models.

### How it works
1. Compute baseline performance (e.g., accuracy, ROC-AUC) on a held-out set.
2. Randomly shuffle (permute) the values of one feature — breaking its relationship with the target while keeping its marginal distribution intact.
3. Recompute performance. The **drop in performance** is that feature's importance.
4. Repeat for each feature (and often multiple shuffles per feature, averaging results for stability).

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc"
)

importances = pd.Series(result.importances_mean, index=X_val.columns).sort_values(ascending=False)
std = pd.Series(result.importances_std, index=X_val.columns)
```

### Key advantages over built-in feature importance
- Works on **any** model (SVM, KNN, neural nets, ensembles).
- Computed on **held-out data**, reflecting genuine predictive value rather than training-set overfitting artifacts.
- Less biased toward high-cardinality features than tree impurity-based importance.

### Caveats
- **Correlated features problem:** if two features are highly correlated, shuffling one doesn't fully break the signal (the model can still "see" it through its correlated partner) — this *understates* importance for correlated groups. Consider grouping correlated features and permuting them together, or removing redundant ones first.
- Computationally more expensive than built-in importance (requires re-scoring many times).
- Importance is relative to the chosen **scoring metric** — different metrics can rank features differently.

---

## 3. SHAP (SHapley Additive exPlanations)

The most theoretically grounded explanation method, based on **Shapley values** from cooperative game theory: fairly distributing a prediction's "payout" among its contributing features.

### The core idea
For any prediction, SHAP answers: *"How much did each feature contribute to pushing the prediction away from the baseline (average) prediction?"*

$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i
$$

Where $\phi_0$ is the baseline (average model output) and each $\phi_i$ is the SHAP value for feature $i$ — how much that feature pushed the prediction up or down for this specific instance.

**Why Shapley values specifically:** they're the *unique* solution satisfying a set of fairness properties (efficiency, symmetry, additivity) for attributing a "team outcome" to individual "players" — features, in this case — considering all possible orderings/coalitions of features.

### Using SHAP in practice
```python
import shap

# TreeExplainer — fast, exact for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Local explanation — why did the model predict this for row 0?
shap.plots.waterfall(shap_values[0])

# Global explanation — aggregate feature importance across all predictions
shap.summary_plot(shap_values, X_test)

# Dependence plot — how does a feature's SHAP value change with its own value?
shap.dependence_plot("age", shap_values, X_test)
```

### Explainer types
| Explainer | Use for |
|---|---|
| `TreeExplainer` | Tree-based models — fast, exact |
| `LinearExplainer` | Linear models — fast, exact |
| `DeepExplainer` / `GradientExplainer` | Neural networks |
| `KernelExplainer` | Any model (model-agnostic) — slow, approximate, uses weighted local linear regression like LIME |

### Reading SHAP plots
- **Waterfall/force plot:** shows how each feature pushed an individual prediction from the baseline to the final output — red = pushes prediction higher, blue = pushes it lower.
- **Summary plot (beeswarm):** every dot is one (sample, feature) SHAP value; color = feature value (red=high, blue=low); position = impact on prediction. Reveals both importance *and* directionality at a glance.
- **Dependence plot:** shows how a single feature's SHAP value varies across its range — reveals non-linear effects and interactions (color by a second feature to see interaction effects).

### Strengths
- Theoretically justified, consistent, both global and local.
- Captures interactions and non-linearities.
- Works across model types.

### Limitations
- `KernelExplainer` is slow for many features/samples.
- Assumes feature independence in some approximations — highly correlated features can produce misleading attributions (the "coalition" of correlated features spreads credit in ways that can be hard to interpret cleanly).
- Computationally expensive for very large datasets/models without the tree/linear-specific fast explainers.

---

## 4. LIME (Local Interpretable Model-agnostic Explanations)

Explains a **single prediction** by approximating the complex model with a simple, interpretable model *locally*, around that specific instance.

### How it works
1. Take the instance you want to explain.
2. Generate perturbed samples around it (slightly modified versions).
3. Get the black-box model's predictions for all those perturbed samples.
4. Fit a simple, interpretable model (usually weighted linear regression) on this local neighborhood, weighting samples by proximity to the original instance.
5. The simple model's coefficients become the explanation — "locally, feature X mattered this much."

```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["negative", "positive"],
    mode="classification"
)

exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)
exp.show_in_notebook()
# or: exp.as_list()  for a plain list of (feature, weight) pairs
```

### SHAP vs. LIME — quick comparison
| | SHAP | LIME |
|---|---|---|
| Theoretical foundation | Game theory (Shapley values), consistent | Heuristic local approximation, no consistency guarantee |
| Speed | Fast for trees/linear models; slow (Kernel) for general models | Generally faster, simpler to compute |
| Scope | Both local and global | Primarily local (per-instance) |
| Stability | More stable/consistent across runs | Can be less stable — perturbation sampling introduces randomness |
| Best for | Rigorous, auditable explanations | Quick, intuitive local explanations, works on text/image data very naturally too |

Both are model-agnostic in their general form; SHAP has model-specific fast implementations that LIME lacks.

---

## 5. Partial Dependence (PDP — Partial Dependence Plot)

A **global** method showing the **average** marginal effect of one (or two) features on the predicted outcome, across the whole dataset.

### How it works
For a feature of interest, PDP:
1. Takes every row in the dataset.
2. Replaces that feature's value with a fixed value (say, age=30) for *every* row, holding all other features at their real, original values.
3. Averages the model's predictions across all rows.
4. Repeats for a range of values across that feature, plotting predicted outcome vs. feature value.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(model, X_train, features=["age", "income"])

# 2D interaction PDP
PartialDependenceDisplay.from_estimator(model, X_train, features=[("age", "income")])
```

### Reading it
- The curve shows how the *average* prediction changes as the feature varies — reveals whether the relationship is linear, monotonic, non-linear, or has a threshold effect.
- A flat line = the model doesn't use that feature much (on average).
- Two-feature PDPs reveal interaction effects (e.g., "age matters more for high-income customers").

### Key assumption & limitation
PDP assumes features are **independent** — when computing the average, it plugs in a fixed value while holding *other* features at their real values, even if that combination is unrealistic (e.g., "house with 10 bedrooms and 200 sqft"). Highly correlated features make PDP potentially misleading, since it can average over combinations that never actually occur in reality.

---

## 6. Individual Conditional Expectation (ICE)

The **per-instance** version of PDP — instead of averaging across all rows, ICE plots a **separate line for every individual instance**, showing how *that specific row's* prediction changes as the feature of interest varies.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_train, features=["age"], kind="individual"  # ICE lines only
)

PartialDependenceDisplay.from_estimator(
    model, X_train, features=["age"], kind="both"  # ICE lines + PDP average overlay
)
```

### Why ICE matters — what PDP can hide
PDP shows only the *average* effect, which can mask **heterogeneous effects**: if a feature increases the prediction for half the population and decreases it for the other half, PDP might show a misleadingly flat average line, while ICE reveals the underlying split into two distinct behavior groups.

**Centered ICE (c-ICE):** shifts every ICE line to start at 0 at the leftmost point, making it easier to compare the *shape* of each instance's curve rather than its absolute level.
```python
PartialDependenceDisplay.from_estimator(
    model, X_train, features=["age"], kind="both", centered=True
)
```

### PDP vs. ICE — when to use which
| | PDP | ICE |
|---|---|---|
| Shows | Average effect across all instances | Effect for each individual instance |
| Reveals interactions/heterogeneity? | No — can mask it | Yes — fanning/crossing lines reveal it |
| Best for | Quick global summary | Detecting whether the global summary is actually representative |
| Typical use | First pass | Follow-up when PDP looks suspicious or flat |

---

## Choosing the Right Explainability Tool

| Question | Recommended tool |
|---|---|
| "Which features matter most overall?" | Feature importance (built-in) or permutation importance |
| "Which features matter, on a model I can't inspect directly (SVM, neural net)?" | Permutation importance |
| "Why did the model predict *this* for *this specific customer*?" | SHAP (waterfall/force plot) or LIME |
| "How does the prediction change as one feature varies, on average?" | Partial Dependence Plot |
| "Does that average effect actually hold for every individual, or does it hide different subgroups?" | ICE |
| "I need a rigorous, theoretically justified, auditable explanation" | SHAP |
| "I need something fast and intuitive for a single prediction, works even on text/images" | LIME |
| "Are two features interacting in how they affect predictions?" | 2D PDP, or SHAP dependence plots colored by a second feature |

### A practical explainability workflow
```python
# 1. Start with built-in or permutation importance for a global overview
result = permutation_importance(model, X_val, y_val, n_repeats=10, scoring="roc_auc")

# 2. Use SHAP summary plot to see global importance + directionality together
shap_values = shap.TreeExplainer(model).shap_values(X_val)
shap.summary_plot(shap_values, X_val)

# 3. Drill into individual predictions of interest (e.g., false positives) with SHAP waterfall or LIME
shap.plots.waterfall(shap_values[misclassified_idx])

# 4. Check top features' relationship shape with PDP, then verify with ICE for heterogeneity
PartialDependenceDisplay.from_estimator(model, X_val, features=["top_feature"], kind="both")
```

---

## Quick Reference

| Method | Scope | Model-agnostic? | Key idea |
|---|---|---|---|
| Built-in feature importance | Global | No | Split-based impurity reduction (trees) or coefficients (linear) |
| Permutation importance | Global | Yes | Performance drop when a feature is shuffled |
| SHAP | Local + Global | Mostly (fast variants are model-specific) | Game-theoretic fair attribution of prediction to features |
| LIME | Local | Yes | Fit a simple local surrogate model around one instance |
| PDP | Global | Yes | Average predicted outcome as one feature varies, others held at real values |
| ICE | Local (per-instance), plotted together | Yes | Same as PDP but one line per instance — reveals heterogeneity PDP hides |

---

## Suggested Learning Path
1. Compare built-in feature importance vs. permutation importance on the same Random Forest — note any features that rank differently.
2. Compute SHAP values for a tree model and generate a summary plot, a waterfall plot for one instance, and a dependence plot for the top feature.
3. Explain the same single prediction with both SHAP and LIME — compare which features each highlights.
4. Plot a PDP for a feature you suspect has heterogeneous effects, then overlay ICE lines and check if the average was hiding a split in behavior.
5. Deliberately include two highly correlated features and observe how permutation importance and PDP both get distorted by the correlation.