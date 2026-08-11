#  Model Training Pipeline

## Learning Objectives

- Structure a reusable training workflow as a single function or script, not ad-hoc notebook cells
- Track experiments (parameters, metrics, artifacts) so results are reproducible and comparable
- Understand what "training workflow" means beyond `model.fit()`: config, logging, versioning

## The Problem

`model.fit(X_train, y_train)` is one line. A training *workflow* answers questions a notebook full of cells doesn't: which exact code and data produced this model? What hyperparameters were used? Can a teammate reproduce this exact result six months from now? Without structure, "which model is the good one" becomes archaeology through notebook history.

## The Concept

### From notebook cells to a training function

A training workflow wraps the whole feature-engineering + model pipeline (Lesson 06) in a function with explicit inputs and outputs:

```python
def train_model(X_train, y_train, config: dict):
    pipeline = build_feature_pipeline(k_best=config["k_best"])
    pipeline.set_params(model__C=config["C"])
    pipeline.fit(X_train, y_train)
    return pipeline
```

The key shift: **everything that could change the result is an explicit, recorded input** (the config dict), not a hardcoded value buried in a cell you might forget to rerun.

### What to track for every training run

| What | Why |
|---|---|
| Config / hyperparameters | So you can reproduce or compare runs |
| Data version (or hash) | Model behavior depends on the exact training data, which changes over time |
| Code version (git commit hash) | So "run #47" maps to an exact snapshot of the pipeline code |
| Metrics (train, validation, test) | So runs are comparable on a consistent basis |
| Artifacts (the fitted pipeline itself, plots, feature importance) | So you can load the actual model later, not just its score |
| Timestamp and environment (library versions) | Silent library version changes can change results |

### A minimal experiment log

Even without a dedicated tool, a structured log is far better than nothing:

```python
import json
import time
from pathlib import Path

def log_run(config, metrics, run_dir="experiments"):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    path = Path(run_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return run_id
```

This alone gives you a directory per run, each with exactly what config produced exactly what metrics — enough to compare five different hyperparameter choices without relying on memory or scrolling through notebook output.

### Experiment tracking tools

For anything beyond a handful of runs, dedicated tools save real time:

| Tool | What it adds |
|---|---|
| MLflow | Logs params/metrics/artifacts to a local or remote tracking server; has a comparison UI |
| Weights & Biases | Hosted tracking with rich dashboards, good for team collaboration |
| DVC | Version control for data and model files, pairs with git for full pipeline reproducibility |

The pattern is the same regardless of tool: define a run, log its config and metrics, save its artifacts, and give it a stable identifier you can look up later.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(config)
    pipeline = train_model(X_train, y_train, config)
    val_score = pipeline.score(X_val, y_val)
    mlflow.log_metric("val_accuracy", val_score)
    mlflow.sklearn.log_model(pipeline, "model")
```

### Training workflow checklist

- Training is a function or script with explicit config in, fitted pipeline out — not a sequence of notebook cells run in an unclear order.
- Every run's config, metrics, and code/data version are logged somewhere durable.
- The fitted pipeline is saved as an artifact (Lesson 09), not just its score.
- Random seeds are set and recorded, so runs are reproducible.
- Comparing two runs means comparing two logged entries, not two mental notes.

See `code/training_pipeline_demo.py` for a runnable minimal training workflow with a lightweight experiment log, building on the feature pipeline from Lesson 06.

## Exercises

1. Wrap the Lesson 06 feature pipeline in a `train_model(config)` function, run it with 3 different `k_best` values, and log each run's config and validation accuracy to its own directory.
2. Write a small script that reads all `experiments/*/metrics.json` files and prints a table sorted by validation accuracy, to simulate a minimal "compare runs" view.
3. Add a random seed to your config, and confirm two runs with the same seed produce identical validation accuracy while two runs with different seeds produce slightly different results.

## Key Terms

| Term | What it actually means |
|---|---|
| Training workflow | The reproducible process of turning a config and training data into a fitted pipeline plus recorded metrics |
| Experiment tracking | Systematically logging the config, metrics, and artifacts of each training run so results are comparable and reproducible |
| Run | One execution of the training workflow with a specific config, data version, and code version, producing one fitted model and its metrics |
| Artifact | A saved output of a run — the fitted model, a plot, a feature importance table — distinct from the metrics that summarize it |
