# Advanced Model Understanding

The final layer beyond fitting and scoring a model: making sure its probabilities mean what they claim, combining multiple models intelligently, quantifying uncertainty, and systematically learning from its mistakes.

---

## 1. Calibration

A model can be *accurate* (correct class predictions) while being **poorly calibrated** — its predicted probabilities don't reflect true likelihoods. Calibration asks: *"When the model says 70% confident, is it actually right about 70% of the time?"*

### Why it matters
- Decisions built on probability thresholds (loan approval cutoffs, medical triage) need probabilities that are *meaningful*, not just well-ranked.
- Some models are notoriously poorly calibrated by default:
  - **Tree-based models & boosted trees** tend to push probabilities toward 0 or 1 (overconfident).
  - **SVMs** don't naturally output probabilities at all (need Platt scaling).
  - **Naive Bayes** tends to produce extreme probabilities due to its independence assumption.
  - **Logistic Regression** is generally well-calibrated by construction (it directly optimizes log loss).

### Diagnosing calibration: Reliability Diagram
Bins predictions by predicted probability, then compares the average predicted probability in each bin to the actual observed positive rate in that bin.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10, strategy="quantile")

plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.legend()
```
- Curve **above** the diagonal → model is **underconfident** in that range.
- Curve **below** the diagonal → model is **overconfident** in that range.

### Fixing calibration

**Platt Scaling** — fits a logistic regression on top of the model's raw scores; works well when the miscalibration is roughly sigmoid-shaped.
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=5)
calibrated_model.fit(X_train, y_train)
```

**Isotonic Regression** — fits a non-parametric, monotonic step function; more flexible, but needs more data to avoid overfitting (works best with larger datasets).
```python
calibrated_model = CalibratedClassifierCV(base_estimator=model, method="isotonic", cv=5)
```

| | Platt Scaling | Isotonic Regression |
|---|---|---|
| Assumes | Sigmoid-shaped miscalibration | No shape assumption |
| Data needed | Works with less data | Needs more data |
| Risk | Underfits complex miscalibration | Can overfit with small data |

### Measuring calibration quantitatively
```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, y_scores)  # lower is better; measures calibration + accuracy together
```
**Brier Score** = mean squared error between predicted probabilities and actual outcomes — a proper scoring rule that rewards both correctness *and* calibration.

> **Golden rule:** always calibrate using a **held-out** set (or via cross-validation, as `CalibratedClassifierCV` does internally) — never calibrate on the same data used to fit the base model.

---

## 2. Probability Prediction

Getting a model to output meaningful probabilities, not just hard class labels — foundational for calibration, thresholding, and uncertainty work.

```python
model.predict(X_test)          # hard class labels
model.predict_proba(X_test)    # probability for each class
```

### How different models produce probabilities
| Model | Native probability output? |
|---|---|
| Logistic Regression | Yes, natively (sigmoid/softmax of linear combination) |
| Naive Bayes | Yes, but often poorly calibrated |
| Decision Trees | Yes — proportion of each class in the leaf node (often crude, few distinct values) |
| Random Forest | Yes — average of individual tree "probabilities" |
| Gradient Boosting / XGBoost | Yes, via sigmoid/softmax on raw scores, but tends toward overconfidence |
| SVM | Not natively — must set `probability=True` (adds Platt scaling internally, and cross-validation cost) |
| Neural Networks | Yes, via softmax/sigmoid output layer — but modern deep networks are often overconfident too |

### Evaluating probability quality (not just the final label)
- **Log loss** — penalizes confident wrong predictions heavily (see Classification Fundamentals guide).
- **Brier score** — mean squared calibration error.
- **Reliability diagrams** — visual calibration check (above).

```python
from sklearn.metrics import log_loss
log_loss(y_test, y_scores)
```

### Multiclass probability output
```python
probs = model.predict_proba(X_test)   # shape: (n_samples, n_classes)
# each row sums to 1 across classes
```
For multiclass calibration, calibration is typically done **per-class** (one-vs-rest), since joint multiclass calibration is more complex.

---

## 3. Ensemble Stacking

Combines predictions from multiple **different** base models using a **meta-model** ("meta-learner") that learns how to best combine them.

### How it works
1. Train several diverse base models (e.g., Random Forest, Gradient Boosting, Logistic Regression, KNN) on the training data.
2. Generate **out-of-fold predictions** from each base model (via cross-validation) — this avoids the meta-model learning from predictions the base models have already "seen" during their own training.
3. Use those out-of-fold predictions as new *features* to train a meta-model (often a simple model like Logistic Regression, to avoid overfitting on top of already-complex base models).
4. At inference time: run new data through all base models, then feed their predictions into the meta-model for the final prediction.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

base_estimators = [
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42)),
    ("svc", SVC(probability=True, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5,               # internally generates out-of-fold predictions to avoid leakage
    stack_method="predict_proba"
)
stacking_model.fit(X_train, y_train)
```

### Why stacking works
Different model types make different kinds of errors — a tree-based model captures non-linear splits well, a linear model captures smooth global trends, KNN captures local structure. The meta-model learns *when to trust which base model*, often outperforming any single base model, especially in competitions (stacking is a Kaggle staple).

### Practical tips
- **Diversity matters more than individual strength** — combining models that make *different* mistakes helps more than combining several near-identical strong models.
- Keep the meta-model simple (regularized linear model) to avoid overfitting on top of already-tuned base models.
- The `cv` parameter inside `StackingClassifier` is essential — without out-of-fold predictions, the meta-model would overfit heavily to base models' training-set performance.

---

## 4. Voting Classifiers

A simpler ensemble method than stacking: combines predictions from multiple models by **voting or averaging**, with no meta-model.

### Hard Voting
Each model votes for a class; the majority class wins.
```python
from sklearn.ensemble import VotingClassifier

hard_voting = VotingClassifier(
    estimators=[("rf", rf_model), ("gb", gb_model), ("lr", lr_model)],
    voting="hard"
)
```

### Soft Voting
Averages the **predicted probabilities** across models, then picks the class with the highest average probability. Generally performs better than hard voting when base models produce reasonably calibrated probabilities, since it uses more information (confidence, not just the final label).
```python
soft_voting = VotingClassifier(
    estimators=[("rf", rf_model), ("gb", gb_model), ("lr", lr_model)],
    voting="soft",
    weights=[2, 1, 1]   # optional — weight more trusted models higher
)
```

### Voting vs. Stacking
| | Voting | Stacking |
|---|---|---|
| Combination rule | Fixed (majority vote / simple average) | Learned by a meta-model |
| Complexity | Simple, fast | More complex, more compute |
| Flexibility | Less — treats all models with a fixed weighting scheme | More — can learn non-linear combination patterns |
| Overfitting risk | Lower | Higher if meta-model isn't kept simple / CV isn't used properly |
| When to use | Quick ensemble boost, models of similar quality | Squeezing out maximum performance, models of varying strengths |

---

## 5. Blending

Similar to stacking, but simpler and less computationally expensive — a common shortcut used in time-constrained settings (e.g., Kaggle competitions with tight deadlines).

### How it differs from stacking
- **Stacking:** base model predictions generated via **k-fold cross-validation** (every training row gets an out-of-fold prediction).
- **Blending:** the training set is split *once* into a training portion and a **holdout** portion. Base models train on the training portion; their predictions on the **holdout** portion become the meta-model's training features.

```python
from sklearn.model_selection import train_test_split

# Split training data into base-train and holdout
X_base, X_holdout, y_base, y_holdout = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Train base models on X_base
rf_model.fit(X_base, y_base)
gb_model.fit(X_base, y_base)

# Generate predictions on the holdout set — these become meta-features
holdout_preds = pd.DataFrame({
    "rf": rf_model.predict_proba(X_holdout)[:, 1],
    "gb": gb_model.predict_proba(X_holdout)[:, 1]
})

# Train meta-model on holdout predictions
meta_model = LogisticRegression().fit(holdout_preds, y_holdout)

# At inference: get base predictions on test data, feed into meta-model
test_preds = pd.DataFrame({
    "rf": rf_model.predict_proba(X_test)[:, 1],
    "gb": gb_model.predict_proba(X_test)[:, 1]
})
final_preds = meta_model.predict(test_preds)
```

### Blending vs. Stacking
| | Blending | Stacking |
|---|---|---|
| Data usage | Single holdout split — some data "wasted" on the holdout, not used to train base models | K-fold — all data used to train base models via out-of-fold rotation |
| Speed | Faster (one split, one set of fits) | Slower (k model fits per base learner) |
| Leakage risk | Low, easy to reason about | Low, but requires careful CV implementation |
| Robustness | Slightly less robust (one split can be unlucky) | More robust (averaged over folds) |
| Common use | Fast iteration, competitions with time limits | Final competitive/production models, when compute allows |

---

## 6. Model Uncertainty

Beyond a single probability, understanding *how confident* the model's confidence itself is — critical for high-stakes decisions.

### Two types of uncertainty
| Type | Meaning | Reducible? |
|---|---|---|
| **Aleatoric uncertainty** | Inherent noise in the data itself (e.g., overlapping classes, measurement noise) | No — irreducible even with more data |
| **Epistemic uncertainty** | Uncertainty from the model's limited knowledge (e.g., sparse training data in a region) | Yes — more data or a better model reduces it |

### Estimating uncertainty in practice

**Ensemble variance (a practical proxy for epistemic uncertainty):**
```python
import numpy as np

# Predictions from multiple models (or multiple bootstrap-trained versions of the same model)
predictions = np.array([model.predict_proba(X_test)[:, 1] for model in ensemble_of_models])
mean_pred = predictions.mean(axis=0)
uncertainty = predictions.std(axis=0)   # high std = high disagreement = high uncertainty
```

**Random Forest — variance across individual trees:**
```python
tree_preds = np.array([tree.predict_proba(X_test)[:, 1] for tree in rf_model.estimators_])
uncertainty = tree_preds.std(axis=0)
```

**Quantile Regression** (for regression uncertainty) — predict multiple quantiles (e.g., 10th, 50th, 90th percentile) to get a prediction interval, not just a point estimate.
```python
from sklearn.ensemble import GradientBoostingRegressor

lower = GradientBoostingRegressor(loss="quantile", alpha=0.1).fit(X_train, y_train)
median = GradientBoostingRegressor(loss="quantile", alpha=0.5).fit(X_train, y_train)
upper = GradientBoostingRegressor(loss="quantile", alpha=0.9).fit(X_train, y_train)
```

**Conformal Prediction** — a model-agnostic framework that produces prediction sets/intervals with a formal statistical coverage guarantee (e.g., "the true label is in this set 90% of the time"), calibrated on a held-out set.
```python
# conceptual sketch using mapie library
from mapie.classification import MapieClassifier
mapie = MapieClassifier(estimator=model, method="score", cv="prefit")
mapie.fit(X_calib, y_calib)
y_pred, y_pred_sets = mapie.predict(X_test, alpha=0.1)  # 90% coverage guarantee
```

**Bayesian approaches** — Bayesian Neural Networks, Gaussian Processes, or MC Dropout (running many stochastic forward passes at inference time with dropout active) produce a full posterior distribution over predictions rather than a single point estimate.

### Why uncertainty matters in practice
- **Selective prediction / abstention** — a model can defer to a human when uncertainty is high, rather than forcing a low-confidence guess.
- **Active learning** — prioritize labeling the most *uncertain* examples to improve the model most efficiently.
- **Risk-aware decisions** — in medicine or finance, knowing "the model isn't sure" is often as important as the prediction itself.

---

## 7. Error Analysis

Beyond aggregate metrics — systematically studying *which* examples the model gets wrong, and *why*, to guide the next round of improvement.

### Step 1: Segment errors
```python
errors = X_test[y_test != y_pred].copy()
errors["true_label"] = y_test[y_test != y_pred]
errors["predicted_label"] = y_pred[y_test != y_pred]
errors["predicted_prob"] = y_scores[y_test != y_pred]
```

**Distinguish error types:**
- **False positives** — inspect for patterns: are certain subgroups over-flagged?
- **False negatives** — inspect for patterns: what commonalities do missed cases share?
- **High-confidence errors** (predicted probability far from 0.5 but wrong) — often the most informative; these reveal systematic blind spots, not just borderline ambiguity.
- **Low-confidence errors** (predicted probability near 0.5) — expected/reasonable; the model was appropriately unsure.

### Step 2: Slice-based analysis
Check performance across meaningful subgroups (demographics, time periods, geographic regions, data sources) — aggregate metrics can hide serious problems in specific slices.
```python
for group in df["segment"].unique():
    mask = df["segment"] == group
    print(group, classification_report(y_test[mask], y_pred[mask]))
```
This is also how **fairness** issues surface — a model with 95% overall accuracy might perform far worse for an underrepresented subgroup.

### Step 3: Look at the confusion matrix by class (multiclass)
Identify which specific classes are most often confused with each other — this often points directly to feature gaps or genuinely ambiguous classes needing better-defined labels.

### Step 4: Residual analysis (regression)
```python
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color="gray", linestyle="--")
```
- **Patterns in residuals vs. predicted value** (e.g., a funnel shape) → heteroscedasticity, suggests the model's errors aren't uniform across the target range.
- **Residuals correlated with a specific feature** → the model isn't capturing that feature's effect well; consider a transform or interaction term.

### Step 5: Manual inspection of worst errors
Sort by largest error magnitude (regression) or highest-confidence-wrong (classification) and manually read through a sample. This is often where the most valuable insight comes from — spotting labeling errors, data quality issues, or a systematic feature gap that no automated metric would surface.

### Step 6: Use explainability tools on the errors specifically
```python
# SHAP on misclassified examples specifically
shap_values_errors = explainer.shap_values(errors_features)
shap.summary_plot(shap_values_errors, errors_features)
```
Comparing SHAP explanations for correct vs. incorrect predictions often reveals *why* the model is failing — e.g., relying too heavily on a feature that's misleading in certain cases.

### Common root causes error analysis reveals
| Symptom | Likely cause |
|---|---|
| Errors cluster in one subgroup | Underrepresented group in training data, or missing feature |
| High-confidence wrong predictions | Model relying on a spurious/misleading correlation |
| Errors on rare classes | Class imbalance not adequately addressed |
| Errors on recent time periods only | Data/concept drift — the world changed since training |
| Errors correlate with a specific feature's missing values | Imputation strategy is hurting that segment |
| Random-looking, no clear pattern | Likely close to irreducible/aleatoric noise — model may already be near its practical ceiling |

---

## Putting It All Together

A mature evaluation process for a model going to production typically includes:

1. **Calibrate** probabilities if the model will drive threshold-based decisions.
2. **Ensemble thoughtfully** (voting/stacking/blending) if squeezing out extra performance matters and compute allows.
3. **Quantify uncertainty** so the system knows when to trust itself less.
4. **Run error analysis** on validation/test errors — segment by subgroup, inspect high-confidence mistakes, and use explainability tools to understand *why* errors happen.
5. **Iterate** — feed error analysis insights back into feature engineering, data collection, or model choice, then re-evaluate.

```python
# Skeleton combining several pieces
calibrated_model = CalibratedClassifierCV(stacking_model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)

y_scores = calibrated_model.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, y_scores)

errors = X_test[y_test != calibrated_model.predict(X_test)]
# ... proceed with slice analysis, SHAP on errors, etc.
```

---

## Quick Reference

| Concept | One-line takeaway |
|---|---|
| Calibration | Predicted probabilities should match real-world frequencies — check with reliability diagrams, fix with Platt/isotonic |
| Probability prediction | Not all models natively produce trustworthy probabilities — evaluate with log loss/Brier score |
| Stacking | Learn how to combine diverse base models via a meta-model trained on out-of-fold predictions |
| Voting | Simple, fixed-rule ensemble combination — hard (majority) or soft (averaged probabilities) |
| Blending | Like stacking but with a single holdout split instead of k-fold — faster, slightly less robust |
| Model uncertainty | Distinguish aleatoric (irreducible) vs. epistemic (reducible) uncertainty; use ensembles, quantile regression, or conformal prediction |
| Error analysis | Systematically study *what* and *why* the model gets wrong — segment, inspect high-confidence errors, use explainability tools |

---

## Suggested Learning Path
1. Plot a reliability diagram for an uncalibrated Gradient Boosting model, then calibrate it and compare Brier scores before/after.
2. Build a `VotingClassifier` (soft) and a `StackingClassifier` on the same base models and compare performance.
3. Implement blending manually with a single holdout split and compare results to the stacking approach.
4. Estimate prediction uncertainty using variance across a Random Forest's individual trees, and identify the most uncertain test examples.
5. Perform full error analysis on a model's test errors: segment by a subgroup, identify the highest-confidence wrong predictions, and use SHAP to explain a few of them.