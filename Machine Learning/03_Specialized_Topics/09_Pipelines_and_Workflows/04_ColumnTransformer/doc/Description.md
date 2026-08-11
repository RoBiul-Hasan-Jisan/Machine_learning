# ColumnTransformer

## Learning Objectives

- Apply different preprocessing to numeric and categorical columns within a single pipeline
- Build a `ColumnTransformer` that routes columns by name or by dtype
- Combine `ColumnTransformer` with `Pipeline` to get one leakage-safe object for a full mixed-type dataset

## The Problem

`Pipeline` (Lesson 03) applies each transformer to the *entire* input. That's fine when every column needs the same treatment, but real tabular data rarely does: numeric columns need scaling and imputation, categorical columns need encoding and a different imputation strategy, and some columns (IDs, free text) may need to be dropped or handled entirely differently.

`ColumnTransformer` applies different transformers to different subsets of columns, then concatenates the results into a single feature matrix — and that whole object is itself a transformer, so it slots straight into a `Pipeline`.

## The Concept

### Routing columns to different pipelines

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ["age", "income", "tenure_months"]
categorical_features = ["region", "plan_type"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])
```

Each entry is `(name, transformer, columns)`. `columns` can be a list of column names (if `X` is a DataFrame), a list of integer positions, a boolean mask, or a selector function (`make_column_selector`, see below). Columns not mentioned in any entry are dropped by default — control this with `remainder`:

```python
preprocessor = ColumnTransformer(
    transformers=[...],
    remainder="passthrough",   # keep unlisted columns unchanged
    # remainder="drop"         # default: drop unlisted columns
)
```

### Selecting columns by dtype instead of by name

Hardcoding column names is brittle if the schema changes. `make_column_selector` selects columns by dtype pattern instead:

```python
from sklearn.compose import make_column_selector

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, make_column_selector(dtype_include="number")),
    ("cat", categorical_pipeline, make_column_selector(dtype_include="object")),
])
```

This is convenient for prototyping, but be careful: it silently changes behavior if a new column's dtype doesn't match your assumption (e.g. a numeric-looking ID column gets scaled like a real numeric feature). For production pipelines, explicit column lists are usually safer and more auditable.

### Nesting inside a full Pipeline

`ColumnTransformer` is itself a transformer, so it becomes step one of a larger `Pipeline` whose final step is the model:

```python
from sklearn.ensemble import GradientBoostingClassifier

full_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", GradientBoostingClassifier()),
])

full_pipeline.fit(X_train, y_train)
full_pipeline.predict(X_test)
```

One `fit`, one `predict`, and every column gets the right treatment, safely refit inside every cross-validation fold.

### Getting output feature names

After one-hot encoding, the number of columns changes and the original names no longer line up. Use `get_feature_names_out()` to recover interpretable names — essential for feature importance plots or debugging:

```python
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()
# e.g. ['num__age', 'num__income', 'cat__region_east', 'cat__region_west', ...]
```

### Common pitfalls

- **Overlapping column lists**: if the same column appears in two transformer entries, it gets processed twice, independently, and both outputs are kept — usually not what you want.
- **`handle_unknown="ignore"` on `OneHotEncoder`**: without this, a category seen at inference time but not at training time raises an error. With it, the unseen category is encoded as all-zeros.
- **Forgetting `remainder`**: columns you didn't explicitly route are silently dropped by default, which is a common source of "why did my accuracy change" bugs after adding a new column.

See `code/column_transformer_demo.py` for a complete mixed-type example with feature name recovery.

## Exercises

1. Build a `ColumnTransformer` for a dataset with 3 numeric and 2 categorical columns, plus one ID column that should be dropped entirely. Verify the ID column does not appear in `get_feature_names_out()`.
2. Set `remainder="passthrough"` and add a boolean column that shouldn't be scaled or encoded. Confirm it passes through unchanged in the output.
3. Introduce a category value in the test set that was never seen in training. Show the pipeline fails without `handle_unknown="ignore"` and succeeds with it.
4. Compare explicit column-name routing vs `make_column_selector(dtype_include="number")` on a DataFrame where one numeric-looking column is actually a categorical ID stored as an integer. Explain the bug this causes.

## Key Terms

| Term | What it actually means |
|---|---|
| ColumnTransformer | An object that applies different transformers to different column subsets and concatenates the results into one feature matrix |
| remainder | The `ColumnTransformer` setting controlling what happens to columns not explicitly routed ("drop" or "passthrough") |
| make_column_selector | A helper that selects columns by dtype pattern rather than by explicit name, for use inside `ColumnTransformer` |
| handle_unknown="ignore" | An `OneHotEncoder` option that encodes categories unseen during training as all-zero, instead of raising an error |
| get_feature_names_out | Method that returns the output column names after transformation, accounting for encoding-driven column expansion |
