#  Model Persistence

## Learning Objectives

- Save and load a fitted pipeline (preprocessing + model) using pickle and joblib
- Explain why the whole pipeline, not just the model, must be persisted
- Know the main risks of persisted models: version mismatches, security, and schema drift

## The Problem

A trained model is useless once your Python process ends unless you save it. And "save the model" almost always means "save the entire fitted `Pipeline`" — the scaler, encoder, feature selector, and model together — because at inference time raw new data needs to go through exactly the same preprocessing, fit with exactly the same learned parameters (means, category mappings, selected features), as it did during training.

## The Concept

### Why persist the whole pipeline, not just the model

If you only save the final estimator and re-write preprocessing by hand at inference time, you risk:

- Using slightly different code (a typo in a manual scaling formula)
- Recomputing statistics (mean, std, category list) from new data instead of reusing the ones learned at training time
- Missing steps entirely (forgetting to one-hot encode a column the same way)

Since `Pipeline` is itself a single fitted object (Lesson 03), saving it saves every step's learned state at once, and loading it restores an object that can immediately call `.predict()` on raw new data.

### Pickle vs joblib

Both serialize Python objects to disk. For scikit-learn objects, `joblib` is generally preferred:

```python
import joblib

joblib.dump(pipeline, "model.joblib")
loaded_pipeline = joblib.load("model.joblib")
```

```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

with open("model.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)
```

| | pickle | joblib |
|---|---|---|
| Standard library | Yes | No (separate package, but a scikit-learn dependency) |
| Efficient with large NumPy arrays | No | Yes — uses memory-mapping, much faster for big arrays |
| scikit-learn recommendation | — | Preferred for scikit-learn models |

For anything with meaningfully sized arrays (most real models), `joblib` is faster to save and load and produces smaller files, because it stores NumPy array data more efficiently than generic pickle.

### What persistence does NOT protect you from

**Library version mismatches.** A pipeline pickled with `scikit-learn==1.3` may fail to load — or worse, load "successfully" but behave subtly differently — under `scikit-learn==1.5`. Always record the exact library versions used to train a persisted model, and pin those versions (or close to them) in the serving environment.

```python
import sklearn
metadata = {
    "sklearn_version": sklearn.__version__,
    "python_version": "3.11",
    "trained_at": "2026-08-11",
    "training_data_hash": "...",
}
```

Save this alongside the model file (e.g. as a companion JSON file) so you can diagnose "it worked in training but not in serving" issues.

**Security.** `pickle.load` (and `joblib.load`, which uses pickle internally) executes arbitrary code embedded in the file. Never load a pickle/joblib file from an untrusted source — treat it the same as running an unreviewed script.

**Schema drift.** If the input schema changes (a column renamed, a new category appears, a column's dtype changes), a loaded pipeline may fail loudly (good) or silently produce nonsense (bad — e.g. `OneHotEncoder` without `handle_unknown="ignore"` would raise, but a renamed column that `ColumnTransformer` can't find will raise a clear error; a column that's present under the right name but with corrupted values will not). Validate incoming data against the expected schema before feeding it to a loaded pipeline (see Lesson 10).

### A practical save/load convention

```python
from pathlib import Path
import joblib, json, sklearn, datetime

def save_pipeline(pipeline, name, metrics, out_dir="models"):
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "pipeline.joblib")
    metadata = {
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.datetime.utcnow().isoformat(),
        "metrics": metrics,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return out

def load_pipeline(path):
    pipeline = joblib.load(Path(path) / "pipeline.joblib")
    with open(Path(path) / "metadata.json") as f:
        metadata = json.load(f)
    return pipeline, metadata
```

This bundles the fitted pipeline with the metadata needed to sanity-check it later — exactly what Lesson 10's inference pipeline will load.

See `code/persistence_demo.py` for a runnable save/load cycle with joblib, including a version-mismatch warning check.

## Exercises

1. Train a pipeline, save it with both `pickle` and `joblib`, and compare the resulting file sizes on a dataset with a few thousand rows.
2. Save a pipeline's metadata alongside the model file, then write a small "load and validate" function that warns if the currently installed `scikit-learn` version doesn't match the one recorded at save time.
3. Simulate schema drift: save a pipeline trained on a DataFrame with columns `[a, b, c]`, then try to predict on a DataFrame missing column `b`. Observe and explain the error.
4. Research (in scikit-learn's docs) the `skops` library, an alternative to pickle/joblib designed to avoid the arbitrary-code-execution risk. Summarize the tradeoff in one paragraph.

## Key Terms

| Term | What it actually means |
|---|---|
| Serialization | Converting a Python object (like a fitted pipeline) into a byte stream that can be saved to disk and reloaded later |
| joblib | A serialization library, preferred over pickle for scikit-learn objects, optimized for large NumPy arrays |
| Version skew | A mismatch between the library versions used to train a model and those used to load/serve it, which can cause silent or loud failures |
| Schema drift | A change in the structure or content of input data over time relative to what a persisted pipeline expects |
| Model metadata | Recorded information about a saved model (library versions, training date, metrics) needed to safely reuse it later |
