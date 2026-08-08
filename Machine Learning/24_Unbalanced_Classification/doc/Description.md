# Imbalanced Classification

Techniques for building and evaluating models when one class vastly outnumbers another — fraud detection, disease diagnosis, churn, defect detection, and similar rare-event problems.

---

## 1. Class Imbalance

### What it is
When one class (usually the "positive"/interesting one) makes up a small fraction of the dataset.

| Imbalance ratio | Example |
|---|---|
| Mild (e.g., 60/40) | Often fine with standard methods |
| Moderate (e.g., 90/10) | Needs some care |
| Severe (e.g., 99/1 or worse) | Fraud detection, rare disease diagnosis, defect detection |

### Why it's a problem
Most ML algorithms are trained to **minimize overall error**, which naturally biases them toward the majority class — a model that always predicts "negative" on a 99/1 dataset gets 99% accuracy while being completely useless. The rare, often more *important*, class gets ignored.

### Detecting it
```python
y.value_counts(normalize=True)
import seaborn as sns
sns.countplot(x=y)
```

### The general toolkit
There's no single fix — imbalance is addressed by combining:
1. **Algorithm-level adjustments** — `class_weight`
2. **Data-level adjustments** — oversampling, undersampling, SMOTE
3. **Evaluation-level adjustments** — the right metrics and curves (PR curve, F1, etc.)

Getting all three right matters more than any single trick.

---

## 2. `class_weight`

The simplest fix: tell the algorithm to **penalize mistakes on the minority class more heavily**, without touching the data itself.

### How it works
Instead of treating every misclassification equally, the loss function is scaled so errors on the minority class cost more — effectively simulating a more balanced dataset during optimization.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 'balanced' automatically weights classes inversely proportional to their frequency
model = LogisticRegression(class_weight="balanced")
model = RandomForestClassifier(class_weight="balanced")
model = SVC(class_weight="balanced", probability=True)
```

`class_weight="balanced"` computes weights as:
$$
w_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_{\text{samples in class } j}}
$$

### Manual weights
```python
model = LogisticRegression(class_weight={0: 1, 1: 10})  # positive class weighted 10x
```
Useful when you know the real-world cost ratio (e.g., missing fraud is 10x worse than a false alarm).

### Other places `class_weight` shows up
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(scale_pos_weight=neg_count / pos_count)  # XGBoost's equivalent

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(class_weight="balanced")
```

**Pros:** no data duplication or loss, computationally cheap, easy to try first.
**Cons:** doesn't help algorithms that don't support weighted loss; may not be enough alone for extreme imbalance.

> **Always try `class_weight` first** — it's the lowest-effort, lowest-risk fix before reaching for resampling.

---

## 3. Random Oversampling

Duplicates examples from the minority class (with replacement) until classes are more balanced.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)  # minority = 50% of majority
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

**Pros:**
- Simple, preserves all original information.
- No information loss (unlike undersampling).

**Cons:**
- Exact duplicates → higher risk of **overfitting** to the specific minority examples that got duplicated.
- Doesn't add any new information/diversity.

> **Critical rule:** oversample only the **training** fold, never the full dataset before splitting — resampling before splitting leaks duplicated rows into validation/test, inflating scores artificially. Always resample inside cross-validation.

---

## 4. Random Undersampling

Removes examples from the majority class until classes are more balanced.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Pros:**
- Reduces dataset size → faster training.
- No duplication artifacts.

**Cons:**
- **Discards potentially useful majority-class data**, which can hurt performance, especially with severe imbalance where the majority class gets cut drastically.

### Smarter undersampling variants
```python
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, NearMiss

# Tomek Links: removes majority-class points that sit right on the decision boundary
tomek = TomekLinks()

# Edited Nearest Neighbours: removes majority samples misclassified by their neighbors
enn = EditedNearestNeighbours()

# NearMiss: keeps majority samples closest to minority samples (informative boundary cases)
nm = NearMiss(version=1)
```

**Oversampling vs. undersampling — quick comparison:**
| | Oversampling | Undersampling |
|---|---|---|
| Dataset size | Grows | Shrinks |
| Information loss | None | Yes (discards majority data) |
| Overfitting risk | Higher (duplicates) | Lower |
| Best when | Small dataset overall | Large dataset, majority class abundant |

---

## 5. SMOTE (Synthetic Minority Oversampling Technique)

Instead of duplicating minority examples exactly, SMOTE **creates new synthetic examples** by interpolating between existing minority-class points and their nearest minority-class neighbors.

### How it works
1. For each minority sample, find its *k* nearest minority-class neighbors.
2. Pick one neighbor at random.
3. Create a new synthetic point along the line connecting the two, at a random position.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Pros over random oversampling:**
- Adds diversity/new synthetic data instead of exact duplicates — reduces overfitting risk.
- Smooths decision boundaries for the minority class.

**Cons:**
- Can create noisy or unrealistic synthetic samples if minority classes overlap heavily with the majority class, or in high-dimensional/sparse feature space.
- Only works on numeric features by default — categorical data needs a variant.
- Like all resampling, must be applied **only to the training fold**, inside cross-validation.

### SMOTE variants
| Variant | What it changes |
|---|---|
| **SMOTE** | Base algorithm, numeric features |
| **Borderline-SMOTE** | Focuses synthetic generation near the decision boundary, where minority classes are most easily misclassified |
| **ADASYN** | Generates more synthetic samples for minority examples that are *harder* to learn (near majority-class regions) |
| **SMOTENC** | Handles a mix of numeric and categorical features |
| **SMOTEN** | For all-categorical data |
| **SMOTE + Tomek / SMOTE + ENN** | Combines oversampling with a cleanup undersampling step to remove noisy/borderline points after synthesis |

```python
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN

# Mixed numeric + categorical data
smote_nc = SMOTENC(categorical_features=[1, 3], random_state=42)

# Combined approach — often the best practical default
smote_tomek = SMOTETomek(random_state=42)
```

### Correct usage inside a Pipeline (critical!)
```python
from imblearn.pipeline import Pipeline as ImbPipeline  # NOTE: use imblearn's Pipeline, not sklearn's
from sklearn.ensemble import RandomForestClassifier

pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(5), scoring="f1")
```
> Using `imblearn.pipeline.Pipeline` (not `sklearn.pipeline.Pipeline`) ensures SMOTE is applied fresh within each CV fold's *training* portion only — never on the validation fold. This is one of the most common leakage mistakes in imbalanced classification.

---

## 6. Precision-Recall Curve

For imbalanced problems, the **PR curve** is usually far more informative than the ROC curve (see Classification Fundamentals guide for full ROC-AUC vs PR-AUC discussion).

### Why PR curve over ROC curve here
ROC's False Positive Rate = FP / (FP + TN). With a huge TN count (common under imbalance), FPR stays small even when precision is actually poor — making ROC curves look artificially good. The PR curve, which never references TN, isn't distorted this way.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
ap = average_precision_score(y_test, y_scores)

plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axhline(y=y_test.mean(), linestyle="--", color="gray", label="No-skill baseline")
plt.legend()
plt.show()
```

**Reading the curve:**
- Top-right corner = ideal (high precision + high recall simultaneously).
- The **no-skill baseline** for PR curves equals the positive class rate (unlike ROC, whose baseline is always the diagonal) — always compare against that, not 0.
- Use the curve to pick an operating threshold matching your business tolerance for false positives vs. false negatives (see Threshold Tuning in the Classification Fundamentals guide).

---

## 7. Appropriate Evaluation Metrics

**Never rely on accuracy alone for imbalanced data.** Use metrics that reflect performance on the minority class specifically.

| Metric | Why it fits imbalanced problems |
|---|---|
| **Precision** | How many flagged positives were real — matters when false alarms are costly |
| **Recall** | How many real positives were caught — matters when missed cases are costly |
| **F1 / Fβ-score** | Balances precision and recall into one number; use Fβ to weight one more heavily |
| **PR-AUC / Average Precision** | Robust ranking metric, unaffected by large TN counts |
| **Balanced Accuracy** | Average of recall on each class — corrects for imbalance, unlike raw accuracy |
| **Matthews Correlation Coefficient (MCC)** | Single balanced summary of the whole confusion matrix, robust even under severe imbalance |
| **Cohen's Kappa** | Measures agreement beyond chance — useful when comparing to a naive baseline |
| **Confusion matrix (raw)** | Always inspect it directly — summary metrics can still hide important patterns |

```python
from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix
)

print(classification_report(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**What to avoid:**
- Plain accuracy as the *primary* metric.
- ROC-AUC as the *only* metric (still fine as a secondary check, just don't rely on it alone).
- Evaluating on a resampled validation set — **always evaluate on the original, untouched distribution** to reflect real-world performance. Only the *training* data should ever be resampled.

---

## Putting It All Together: A Practical Workflow

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, average_precision_score

# 1. Split first, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Try class_weight first — cheapest fix
baseline = RandomForestClassifier(class_weight="balanced", random_state=42)

# 3. Try resampling (SMOTE) inside a proper pipeline, only on training folds
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="average_precision")
print("CV PR-AUC:", scores.mean(), "+/-", scores.std())

# 4. Fit on full training data, evaluate ONCE on original (untouched) test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_scores = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("Test PR-AUC:", average_precision_score(y_test, y_scores))
```

### Decision guide: which fix to reach for
| Situation | Recommended approach |
|---|---|
| Mild imbalance, want a quick fix | `class_weight="balanced"` |
| Small dataset overall | Oversampling (SMOTE preferred over random duplication) |
| Large dataset, majority class abundant | Undersampling (or combine with SMOTE) |
| Severe/extreme imbalance | Combine `class_weight` + SMOTE + threshold tuning + PR-AUC evaluation |
| Any imbalance level | Always evaluate with precision/recall/F1/PR-AUC, never accuracy alone |

---

## Quick Reference

| Concept | One-line takeaway |
|---|---|
| Class imbalance | Standard training/metrics silently favor the majority class |
| `class_weight` | Cheapest fix — penalize minority-class errors more in the loss function |
| Random oversampling | Duplicates minority rows — simple but risks overfitting |
| Random undersampling | Drops majority rows — simple but loses information |
| SMOTE | Generates synthetic minority samples via interpolation — better diversity than duplication |
| PR curve | The right curve for imbalanced problems — unaffected by large TN counts |
| Right metrics | Precision, Recall, F1, PR-AUC, Balanced Accuracy, MCC — never accuracy alone |
| Golden rule | Resample only training folds; always evaluate on the original, untouched distribution |

---

## Suggested Learning Path
1. Create a synthetic imbalanced dataset (`sklearn.datasets.make_classification` with `weights=[0.95, 0.05]`) and check accuracy vs. F1 for a naive baseline.
2. Train the same model with and without `class_weight="balanced"` and compare recall on the minority class.
3. Apply random oversampling, random undersampling, and SMOTE separately (each inside a proper CV pipeline) and compare PR-AUC.
4. Plot ROC and PR curves side by side for the same model on an imbalanced dataset — observe how much better ROC looks vs. PR.
5. Deliberately resample *before* splitting to see how it inflates validation scores — then fix it by resampling inside the pipeline only.