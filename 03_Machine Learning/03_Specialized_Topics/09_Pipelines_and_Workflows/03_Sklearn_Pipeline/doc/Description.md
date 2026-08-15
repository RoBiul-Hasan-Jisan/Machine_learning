#  Sklearn Pipeline

## Learning Objectives

- Explain why chaining preprocessing and modeling into one object prevents leakage and simplifies deployment
- Build, fit, and predict with `sklearn.pipeline.Pipeline`
- Access, inspect, and tune individual steps inside a fitted pipeline

## The Problem

A typical model needs several steps before `fit`: scale the features, maybe impute missing values, maybe select features, then fit the estimator. Doing this as separate script steps is fragile:

- You must remember to apply the exact same transformations, fit on train only, at both training and inference time.
- Cross-validation becomes error-prone: if you scale before splitting into folds, you leak fold statistics across the split (see Lesson 02).
- Saving "the model" for deployment actually means saving several objects (scaler, imputer, encoder, model) and re-applying them in the right order — easy to get wrong.

`Pipeline` solves all three: it bundles a sequence of steps into a single estimator-like object with one `fit` / `predict`, and it plays correctly with cross-validation and model persistence.

## The Concept

### Pipeline as a chain of transformers ending in an estimator

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier()),
])
```

Every step except the last must implement `fit` and `transform` (a "transformer"). The last step just needs `fit` (and usually `predict`) — it's the estimator. Calling `pipe.fit(X_train, y_train)` calls `fit_transform` on each transformer in order, then `fit` on the final estimator using the fully transformed data. Calling `pipe.predict(X_test)` calls `transform` (not `fit_transform`) on each transformer, then `predict` on the estimator.

```
fit(X_train, y_train):
    X1 = scaler.fit_transform(X_train)
    model.fit(X1, y_train)

predict(X_test):
    X1 = scaler.transform(X_test)      # note: transform only, no fit
    return model.predict(X1)
```

This is exactly the leakage-safe order from Lesson 02, enforced automatically. It's also why `Pipeline` works correctly inside `cross_val_score` and `GridSearchCV`: each fold refits the scaler on that fold's training portion only.

### Why not just write the steps by hand?

You could. But `Pipeline` gives you three things a hand-rolled sequence doesn't:

1. **One object to pass around.** `cross_val_score(pipe, X, y, cv=5)` correctly refits scaling inside every fold. Doing this by hand means writing your own CV loop and remembering to refit preprocessing each time.
2. **One object to save.** `joblib.dump(pipe, "model.pkl")` saves the scaler and the model together, in the right order (Lesson 09).
3. **Hyperparameter tuning across the whole chain.** `GridSearchCV` can tune the model's hyperparameters and a preprocessing step's hyperparameters (e.g. `PCA(n_components=...)`) in the same search, using step-name-prefixed parameter keys (Lesson 08).

### Naming steps and accessing them

Each step gets a string name, used to access it later or to specify hyperparameters for tuning:

```python
pipe.named_steps["scaler"]          # access the fitted StandardScaler
pipe.named_steps["model"]           # access the fitted RandomForestClassifier
pipe["model"].feature_importances_  # equivalent shorthand indexing
```

For hyperparameter tuning, parameter names use `stepname__paramname`:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [None, 5, 10],
}
search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
search.fit(X_train, y_train)
```

`make_pipeline` is a shortcut that auto-generates step names from the class name (lowercased):

```python
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
# step names become "standardscaler" and "randomforestclassifier"
```

Use `Pipeline` with explicit names when you need predictable, readable names for tuning or logging. Use `make_pipeline` for quick, throwaway chains.

### Inspecting a fitted pipeline

```python
pipe.steps                 # list of (name, estimator) tuples
pipe.named_steps           # dict access by name
pipe[:-1].transform(X)     # apply everything except the final estimator (get transformed features)
pipe[-1]                   # the final estimator
```

`pipe[:-1].transform(X)` is a useful debugging trick: it runs the full preprocessing chain and stops before the model, so you can inspect exactly what the model actually sees.

See `code/pipeline_demo.py` for a complete example: building, fitting, inspecting, and tuning a pipeline on a real dataset.

## Exercises

1. Build a pipeline with `StandardScaler` and `LogisticRegression`. Fit it, then use `pipe[:-1].transform(X_test)` to print the first 3 rows of transformed features and confirm they are zero-mean, unit-variance.
2. Compare `cross_val_score` on the pipeline vs a hand-written loop that scales the whole dataset once before splitting into folds. Show the hand-written version gives an over-optimistic score on a dataset where scaling parameters matter.
3. Add a third step (e.g. `PCA(n_components=5)`) between the scaler and the model. Confirm `predict` still works end-to-end with one call.
4. Use `GridSearchCV` to jointly tune `PCA__n_components` and `model__max_depth` in a 3-step pipeline.

## Key Terms

| Term | What it actually means |
|---|---|
| Pipeline | A chain of transformers followed by an estimator, exposed as a single object with `fit`/`predict` |
| Transformer | An object implementing `fit` and `transform` (e.g. a scaler, imputer, or encoder) |
| Estimator | An object implementing `fit` and typically `predict`; the final step of a pipeline |
| `named_steps` | Dictionary access to individual fitted steps inside a pipeline, by their string name |
| Step-name-prefixed parameters | The `stepname__paramname` convention used by `GridSearchCV` to tune parameters of any step in a pipeline |
