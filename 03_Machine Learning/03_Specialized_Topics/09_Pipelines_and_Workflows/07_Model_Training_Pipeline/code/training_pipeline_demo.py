

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_fake_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(40, 12, n).round(1),
        "income": rng.normal(60000, 15000, n).round(0),
        "tenure_months": rng.integers(0, 72, n),
        "region": rng.choice(["east", "west", "north", "south"], n),
        "plan_type": rng.choice(["basic", "premium"], n),
    })
    logit = (
        0.03 * (df["age"] - 40)
        + 0.00004 * (df["income"] - 60000)
        + 0.02 * (df["tenure_months"] - 36)
        + (df["plan_type"] == "premium") * 0.8
        + rng.normal(0, 1.2, n)
    )
    y = (logit > 0).astype(int)
    return df, y


def build_pipeline(config):
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

    return Pipeline([
        ("preprocessing", preprocessing),
        ("select", SelectKBest(score_func=f_classif, k=config["k_best"])),
        ("model", LogisticRegression(C=config["C"], max_iter=1000,
                                      random_state=config["seed"])),
    ])


def train_model(X_train, y_train, config):
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    return pipeline


def log_run(config, metrics, run_dir="experiments"):
    run_id = time.strftime("%Y%m%d_%H%M%S_") + f"k{config['k_best']}_C{config['C']}"
    path = Path(run_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return run_id


def compare_runs(run_dir="experiments"):
    rows = []
    for run_path in sorted(Path(run_dir).glob("*")):
        with open(run_path / "config.json") as f:
            config = json.load(f)
        with open(run_path / "metrics.json") as f:
            metrics = json.load(f)
        rows.append({**config, **metrics, "run_id": run_path.name})
    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    return df


def main():
    df, y = make_fake_data()
    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    for k_best in [3, 5]:
        for C in [0.1, 1.0]:
            config = {"k_best": k_best, "C": C, "seed": 42}
            pipeline = train_model(X_train, y_train, config)
            metrics = {
                "train_accuracy": round(pipeline.score(X_train, y_train), 4),
                "val_accuracy": round(pipeline.score(X_val, y_val), 4),
            }
            run_id = log_run(config, metrics)
            print(f"Run {run_id}: {metrics}")

    print("\n=== Comparison across runs ===")
    print(compare_runs()[["run_id", "k_best", "C", "train_accuracy", "val_accuracy"]])


if __name__ == "__main__":
    main()
