
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_COLUMNS = ["age", "income", "tenure_months", "region", "plan_type"]
VALID_REGIONS = {"east", "west", "north", "south"}


# Training side (would normally live in a separate training script)

def make_fake_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(40, 12, n).round(1),
        "income": rng.normal(60000, 15000, n).round(0),
        "tenure_months": rng.integers(0, 72, n),
        "region": rng.choice(list(VALID_REGIONS), n),
        "plan_type": rng.choice(["basic", "premium"], n),
    })
    logit = (
        0.03 * (df["age"] - 40)
        + 0.00004 * (df["income"] - 60000)
        + (df["plan_type"] == "premium") * 0.8
        + rng.normal(0, 1.2, n)
    )
    y = (logit > 0).astype(int)
    return df, y


def train_and_save(model_dir="models/churn_model"):
    df, y = make_fake_data()
    X_train, _, y_train, _ = train_test_split(df, y, test_size=0.2, random_state=42)

    preprocessing = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), ["age", "income", "tenure_months"]),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), ["region", "plan_type"]),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, Path(model_dir) / "pipeline.joblib")
    return model_dir


# Inference side

def validate_input(raw_input: dict):
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_input]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    age = raw_input.get("age")
    if age is not None and not (0 < age < 120):
        raise ValueError(f"age out of expected range: {age}")

    region = raw_input.get("region")
    if region is not None and region not in VALID_REGIONS:
        raise ValueError(f"Unrecognized region: {region}")


def load_model(model_dir):
    return joblib.load(Path(model_dir) / "pipeline.joblib")


def predict_one(raw_input: dict, pipeline) -> dict:
    validate_input(raw_input)
    X = pd.DataFrame([raw_input])
    prediction = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0].tolist()
    return {"prediction": prediction, "probability": proba}


def batch_predict(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    out = df.copy()
    out["prediction"] = pipeline.predict(df)
    out["probability_positive"] = pipeline.predict_proba(df)[:, 1]
    return out


def demo_load_once_vs_reload_per_call(model_dir, n_calls=50):
    sample = {"age": 35, "income": 55000, "tenure_months": 24, "region": "east", "plan_type": "basic"}

    start = time.perf_counter()
    for _ in range(n_calls):
        pipeline = load_model(model_dir)   # BAD: reloading from disk every call
        predict_one(sample, pipeline)
    reload_time = time.perf_counter() - start

    pipeline = load_model(model_dir)       # GOOD: load once
    start = time.perf_counter()
    for _ in range(n_calls):
        predict_one(sample, pipeline)
    reuse_time = time.perf_counter() - start

    print(f"Reload-per-call: {reload_time:.4f}s for {n_calls} calls")
    print(f"Load-once-reuse: {reuse_time:.4f}s for {n_calls} calls")
    print(f"Speedup: {reload_time / reuse_time:.1f}x")


def main():
    model_dir = train_and_save()
    pipeline = load_model(model_dir)

    print("=== Single-row inference ===")
    sample = {"age": 29, "income": 48000, "tenure_months": 6, "region": "west", "plan_type": "premium"}
    print(predict_one(sample, pipeline))

    print("\n=== Invalid input handling ===")
    bad_sample = {"age": 200, "income": 48000, "tenure_months": 6, "region": "west", "plan_type": "premium"}
    try:
        predict_one(bad_sample, pipeline)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n=== Batch inference ===")
    batch_df, _ = make_fake_data(n=10, seed=99)
    result = batch_predict(batch_df, pipeline)
    print(result[["age", "region", "plan_type", "prediction", "probability_positive"]])

    print("\n=== Load-once vs reload-per-call ===")
    demo_load_once_vs_reload_per_call(model_dir)


if __name__ == "__main__":
    main()
