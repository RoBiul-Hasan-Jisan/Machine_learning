

import datetime
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    path = Path(path)
    pipeline = joblib.load(path / "pipeline.joblib")
    with open(path / "metadata.json") as f:
        metadata = json.load(f)

    if metadata["sklearn_version"] != sklearn.__version__:
        print(
            f"WARNING: model was saved with scikit-learn "
            f"{metadata['sklearn_version']}, but the current environment has "
            f"{sklearn.__version__}. Behavior may differ."
        )
    else:
        print(f"scikit-learn version matches ({sklearn.__version__}). OK to use.")

    return pipeline, metadata


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"Trained pipeline test accuracy: {test_acc:.4f}")

    out_dir = save_pipeline(pipeline, "breast_cancer_logreg", {"test_accuracy": test_acc})
    print(f"Saved pipeline + metadata to: {out_dir}\n")

    loaded_pipeline, metadata = load_pipeline(out_dir)
    print("Loaded metadata:", metadata)

    # Confirm loaded pipeline produces identical predictions
    original_preds = pipeline.predict(X_test)
    loaded_preds = loaded_pipeline.predict(X_test)
    assert (original_preds == loaded_preds).all()
    print("\nLoaded pipeline predictions match the original exactly.")


if __name__ == "__main__":
    main()
