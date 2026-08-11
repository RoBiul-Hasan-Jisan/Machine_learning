#  Custom Transformers

## Learning Objectives

- Build a custom transformer that follows the scikit-learn API contract
- Understand why inheriting from `BaseEstimator` and `TransformerMixin` matters for pipeline compatibility
- Know the difference between logic that belongs in `fit` vs `transform`

## The Problem

Built-in transformers (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`, ...) cover generic cases. Real projects need domain-specific logic: extracting the day-of-week from a timestamp, computing a ratio between two columns, clipping outliers using a threshold learned from training data, or applying a business rule. If that logic doesn't follow the same `fit`/`transform` contract as built-in transformers, it can't be dropped into a `Pipeline` or `ColumnTransformer`, and you're back to writing it by hand outside the pipeline — with all the leakage and reproducibility risk from Lessons 02-03.

## The Concept

### The scikit-learn transformer contract

Any object with `fit(X, y=None)` returning `self`, and `transform(X)` returning the transformed data, can be used as a pipeline step. Inheriting from two mixin classes gives you useful behavior for free:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        self.param = param          # store args unchanged, no logic here

    def fit(self, X, y=None):
        # learn anything the transform step needs from TRAINING data only
        self.learned_value_ = ...   # trailing underscore = "fitted attribute"
        return self

    def transform(self, X):
        # apply the transformation, using self.learned_value_
        X_transformed = ...
        return X_transformed
```

- `BaseEstimator` gives you `get_params()` / `set_params()` for free, based on your `__init__` signature. This is what makes `GridSearchCV` able to tune your custom transformer's hyperparameters with `stepname__param` syntax.
- `TransformerMixin` gives you `fit_transform()` for free, implemented as `fit(X).transform(X)` (with an optimized path where relevant).

### Rules that keep a custom transformer well-behaved

1. **`__init__` should only store arguments, unchanged.** No validation, no computation. This is a scikit-learn convention `get_params()`/`set_params()`/cloning rely on — put logic in `fit` instead.
2. **Any statistic learned from data goes in `fit`, not `transform`.** If you compute a mean, a threshold, or a fitted encoder inside `transform`, you are refitting on whatever data you're given — including the test set — which reintroduces the exact leakage Lesson 02 warned about.
3. **Fitted attributes get a trailing underscore** (`self.mean_`, `self.threshold_`) by convention, so it's visually clear which attributes only exist after `fit` has been called.
4. **`transform` must not mutate its input.** Return a new object; don't modify `X` in place.
5. **`transform` should work on any dataset with the same schema, not just the training set.** Since it will be called at inference time on new data.

### Example: an outlier clipper fitted on training data

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips each column to [Q1 - k*IQR, Q3 + k*IQR], learned from training data."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)
```

Because the clip bounds (`lower_`, `upper_`) are learned in `fit` from the training fold only, this transformer is safe to use inside `cross_val_score` or `GridSearchCV` — each fold learns its own bounds from its own training portion.

### Example: a feature-engineering transformer using domain logic

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RatioFeature(BaseEstimator, TransformerMixin):
    """Adds a new column: numerator / denominator, with a safe fallback for zero."""

    def __init__(self, numerator_col, denominator_col, output_name="ratio"):
        self.numerator_col = numerator_col
        self.denominator_col = denominator_col
        self.output_name = output_name

    def fit(self, X, y=None):
        return self  # nothing to learn from data here

    def transform(self, X):
        X = X.copy()
        denom = X[self.denominator_col].replace(0, pd.NA)
        X[self.output_name] = X[self.numerator_col] / denom
        return X
```

Not every transformer needs to learn something in `fit` — some just apply a fixed, stateless rule. `fit` still needs to exist and return `self` so the object satisfies the contract and can sit inside a `Pipeline`.

### Using `FunctionTransformer` for simple, stateless cases

For quick, stateless transformations, you don't need a full class — `FunctionTransformer` wraps a plain function:

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transformer = FunctionTransformer(np.log1p, validate=True)
```

Reach for a full `BaseEstimator`/`TransformerMixin` class as soon as the transformation needs to learn anything from training data (thresholds, means, fitted sub-encoders) or needs tunable hyperparameters.

See `code/custom_transformer_demo.py` for both examples above, used inside a full `Pipeline` and verified to behave correctly under cross-validation.

## Exercises

1. Implement `OutlierClipper` (above) and confirm that clip bounds computed on a training fold differ from bounds computed on the full dataset, when the data has skewed outliers concentrated in one fold.
2. Write a custom transformer that adds a `days_since_signup` feature from a `signup_date` column and today's date. Discuss why "today's date" is a subtle leakage risk if the model is retrained at a different time than it's deployed.
3. Wrap `RatioFeature` and `OutlierClipper` together in one `Pipeline`, and confirm `get_params()` exposes tunable hyperparameters (like `k`) with `GridSearchCV`-compatible names.
4. Convert a stateless custom transformer into a `FunctionTransformer` one-liner and confirm both produce identical output.

## Key Terms

| Term | What it actually means |
|---|---|
| BaseEstimator | Scikit-learn mixin providing `get_params`/`set_params`, required for a custom object to work with `GridSearchCV` and cloning |
| TransformerMixin | Scikit-learn mixin providing `fit_transform` for free, given `fit` and `transform` |
| Fitted attribute | An attribute (conventionally named with a trailing underscore) that only exists after `fit` has been called, e.g. `self.mean_` |
| FunctionTransformer | A wrapper that turns a plain stateless function into a valid pipeline transformer, without writing a full class |
| Stateless vs stateful transformer | Whether `transform` depends only on the current input (stateless) or on statistics learned during `fit` (stateful) |
