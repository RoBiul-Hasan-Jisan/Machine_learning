# Feature Engineering Pipeline

## Learning Objectives

- Assemble imputation, encoding, scaling, and feature selection into one ordered pipeline
- Choose appropriate strategies for each step based on data type and downstream model
- Understand why feature selection must be fit inside the pipeline, not before it

## The Problem

Lessons 03-05 covered the pieces: `Pipeline`, `ColumnTransformer`, and custom transformers. Feature engineering in practice is a specific, ordered combination of these pieces, applied consistently: fill in missing values, turn categories into numbers, put numeric features on comparable scales, and optionally cut down to the most useful features. Get the order or the fitting boundary wrong, and you either leak information or feed the model garbage.

## The Concept

### The standard order

```
Raw Features
     ↓
Imputation        (fill missing values)
     ↓
Encoding          (categories → numbers)
     ↓
Scaling           (put numeric features on comparable ranges)
     ↓
Feature Selection (keep only the most useful features)
     ↓
Model-ready Features
```

Why this order:
- **Imputation before encoding**: you can't one-hot encode a missing value sensibly without deciding first whether it's its own category or should be imputed.
- **Encoding before scaling**: one-hot encoded columns are already 0/1, so scaling is usually skipped for them; but if you scale before encoding, you'd be scaling raw category codes, which is meaningless.
- **Scaling before feature selection**: several feature selection methods (e.g. L1-based, distance-based) are scale-sensitive, so scaling should happen first for a selection step to work correctly.

### Imputation strategies

| Strategy | When to use |
|---|---|
| `mean` / `median` | Numeric features; median is more robust to outliers |
| `most_frequent` | Categorical features, or numeric features that are effectively discrete |
| `constant` (with a fill value like `"missing"` or `-1`) | When "missingness" itself may be informative and you want the model to be able to learn from it |
| Model-based (`IterativeImputer`, KNN imputer) | When missingness is not random and simple statistics under-use available information |

Always check whether missingness is random or informative before picking a strategy — missing *not at random* data (e.g. a survey question skipped because of the answer) can carry signal that a naive mean-fill destroys.

### Encoding strategies

| Strategy | When to use |
|---|---|
| One-hot (`OneHotEncoder`) | Low-to-moderate cardinality categoricals, no ordinal relationship |
| Ordinal (`OrdinalEncoder`) | Categories with a genuine order (e.g. "low"/"medium"/"high"), or as a fast baseline for tree models |
| Target encoding | High-cardinality categoricals (e.g. zip code), where one-hot would explode dimensionality — but requires care to avoid leakage (fit only on training folds) |
| Frequency/count encoding | High-cardinality categoricals where the count itself is a useful signal |

Tree-based models (random forests, gradient boosting) can work reasonably with ordinal-encoded categories even without true order, since trees can split arbitrarily on the encoded values. Linear models generally need one-hot or a leakage-safe target encoding to avoid inventing a false ordinal relationship.

### Scaling strategies

| Strategy | When to use |
|---|---|
| `StandardScaler` (zero mean, unit variance) | Default choice; needed for linear models, SVMs, neural nets, PCA |
| `MinMaxScaler` (scale to [0, 1]) | When you need bounded ranges (e.g. some neural net activations) |
| `RobustScaler` (uses median/IQR) | Data with significant outliers |
| None | Tree-based models generally don't need scaling — splits are invariant to monotonic transformations |

### Feature selection strategies

| Strategy | When to use |
|---|---|
| Variance threshold | Remove near-constant features early, cheaply |
| Univariate (`SelectKBest` with a statistical test) | Fast, model-agnostic first pass |
| Model-based (`SelectFromModel` using L1 or tree importances) | Selection informed by a model, often stronger than univariate |
| Recursive feature elimination (`RFE`) | Smaller feature sets where the extra compute cost is affordable |

**Critical**: feature selection must be fit only on the training fold, exactly like scaling. If you select features using the full dataset's correlation with the target, you've used test-set labels to influence what the model sees during training — a subtle but real form of leakage. Wrapping selection inside the `Pipeline` fixes this the same way it fixes scaling leakage.

### Putting it together

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

numeric_features = ["age", "income", "tenure_months"]
categorical_features = ["region", "plan_type"]

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessing = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

feature_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("select", SelectKBest(score_func=f_classif, k=10)),
    ("model", LogisticRegression(max_iter=1000)),
])
```

One `fit`, one `predict`, every step scoped correctly to training folds. This is the pattern Lesson 07 builds a full training workflow around.

See `code/feature_pipeline_demo.py` for a runnable end-to-end version, including a before/after comparison showing the leakage risk of selecting features outside the pipeline.

## Exercises

1. Build the full pipeline above on a dataset with missing values in both numeric and categorical columns. Confirm predictions work end-to-end on new data containing a missing value pattern never seen during training.
2. Compare `SelectKBest` fit inside the pipeline (correct) vs fit once on the full dataset before splitting (leaky). Measure the difference in cross-validated accuracy.
3. Swap `OneHotEncoder` for `OrdinalEncoder` on a tree-based model and compare accuracy and training time to the one-hot version.
4. Add a `VarianceThreshold` step before `SelectKBest` and confirm it removes any near-constant engineered features.

## Key Terms

| Term | What it actually means |
|---|---|
| Imputation | Filling in missing values using a learned or fixed strategy |
| Missing not at random | Missingness that itself carries information about the target, as opposed to missingness that's effectively random |
| Target encoding | Replacing a categorical value with a statistic (e.g. mean target) computed from training data for that category; leakage-prone if not fit per-fold |
| SelectKBest | A feature selection transformer that keeps the k highest-scoring features according to a statistical test |
| SelectFromModel | A feature selection transformer that keeps features a fitted model considers important (e.g. nonzero L1 coefficients) |
