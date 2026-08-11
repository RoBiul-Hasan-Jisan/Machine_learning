#  Inference Pipeline

## Learning Objectives

- Build a standalone inference path that loads a persisted pipeline and serves predictions on raw input
- Validate incoming data against the schema the pipeline expects, before prediction
- Distinguish batch inference from real-time (online) inference and know when to use each

## The Problem

Training produces a fitted pipeline. Inference is a separate concern: given one new row (or a batch of rows) of *raw* input — the same shape and types as the original raw training data, before any preprocessing — produce a prediction, reliably, without re-deriving any of the training-time logic by hand.

```
Raw Input
    ↓
Preprocessing   (the SAME fitted steps from training — no refitting)
    ↓
Feature Engineering
    ↓
Model
    ↓
Prediction
```

The critical property: every step in this diagram is exactly the fitted pipeline from training (Lesson 09), called with `.transform()`/`.predict()`, never `.fit()` again. Inference code should never re-learn a mean, a category list, or a selected feature set from the incoming data — it only applies what was learned at training time.

## The Concept

### The inference function

```python
def predict(raw_input: dict, pipeline) -> dict:
    import pandas as pd
    X = pd.DataFrame([raw_input])          # one row -> one-row DataFrame
    prediction = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0].tolist() if hasattr(pipeline, "predict_proba") else None
    return {"prediction": prediction, "probability": proba}
```

This function assumes `raw_input` matches the schema the pipeline's `ColumnTransformer` was fit on — same column names, same rough value ranges. That assumption needs to be checked, not hoped for.

### Schema validation before prediction

Catch bad input before it reaches the model, with a clear error, rather than letting it fail deep inside a transformer (or silently produce a nonsense prediction).

```python
REQUIRED_COLUMNS = ["age", "income", "tenure_months", "region", "plan_type"]

def validate_input(raw_input: dict):
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_input]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if raw_input["age"] is not None and not (0 < raw_input["age"] < 120):
        raise ValueError(f"age out of expected range: {raw_input['age']}")

    if raw_input["region"] not in {"east", "west", "north", "south", None}:
        raise ValueError(f"Unrecognized region: {raw_input['region']}")
```

This is a lightweight, hand-written version of what tools like `pydantic` (for API request validation) or `pandera`/`great_expectations` (for DataFrame-level schema and data-quality checks) do more thoroughly at scale. The principle is the same regardless of tool: define what "valid input" means once, check it at the boundary, fail with a clear message instead of a confusing downstream stack trace.

### Batch vs real-time inference

| | Batch inference | Real-time (online) inference |
|---|---|---|
| Trigger | Scheduled job (hourly, daily) | A single request needs a response |
| Input | Many rows at once (a table, a file) | Usually one row (or a small handful) per call |
| Latency requirement | Minutes to hours is often fine | Milliseconds to low seconds |
| Typical serving shape | A script or job that loads the pipeline once, predicts on a DataFrame, writes results out | An API endpoint that loads the pipeline once at startup, predicts per request |
| Example | Score all customers overnight for a churn dashboard | Fraud check on a transaction as it happens |

**Batch inference** pattern:

```python
def batch_predict(input_path, output_path, pipeline):
    df = pd.read_csv(input_path)
    predictions = pipeline.predict(df)
    df["prediction"] = predictions
    df.to_csv(output_path, index=False)
```

**Real-time inference** pattern (a minimal API, using any web framework):

```python
# pseudocode - framework-agnostic shape
pipeline, metadata = load_pipeline("models/latest")   # load ONCE at startup

def handle_request(raw_json):
    validate_input(raw_json)
    return predict(raw_json, pipeline)                # called per request, no reload
```

The shared rule in both: load the persisted pipeline **once**, and reuse the in-memory object for every prediction — reloading it from disk per request (or per row) is a common, easily avoidable performance bug.

### Preprocessing consistency: the whole point

Because the pipeline persisted in Lesson 09 already contains the fitted `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, and `SelectKBest`, `pipeline.predict(new_raw_df)` runs the exact same transformations, with the exact same learned parameters, as during training — a missing `age` gets the *training-set* median, `region="east"` gets the *training-fit* one-hot mapping, and the selected features are the *training-selected* ones. This is the entire payoff of Lessons 03-06: inference code stays this short precisely because none of that logic needs to be reimplemented.

See `code/inference_pipeline_demo.py` for a runnable example: train + save a pipeline, then a separate inference path that loads it, validates input, and serves both single-row and batch predictions.

## Exercises

1. Build the `validate_input` function for a dataset of your choice, covering required fields, type checks, and range checks. Confirm it raises a clear error for each invalid case you can think of.
2. Implement `batch_predict` and run it on a CSV with 500 rows, then confirm the output row count and column set match expectations.
3. Simulate a "reload per request" performance bug: time 100 predictions with the pipeline reloaded from disk each time vs loaded once and reused. Report the speedup.
4. Extend `predict` to include an "unknown category" flag using `pipeline.named_steps["preprocessing"]` internals, so downstream consumers know when a prediction relied on `handle_unknown="ignore"` filling in zeros for a never-seen category.

## Key Terms

| Term | What it actually means |
|---|---|
| Inference | Using a trained, persisted pipeline to produce predictions on new, raw data |
| Schema validation | Checking that incoming data matches the structure and value ranges a pipeline expects, before running prediction |
| Batch inference | Producing predictions for many rows at once, typically on a schedule, where latency per-row is not critical |
| Real-time (online) inference | Producing a prediction for a single request with low-latency requirements, typically behind an API |
| Load once, predict many | The pattern of loading a persisted pipeline into memory a single time and reusing it across many predictions, rather than reloading per call |
