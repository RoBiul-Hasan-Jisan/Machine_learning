

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_fake_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(n),                                  # should be dropped
        "age": rng.normal(40, 12, n).round(1),
        "income": rng.normal(60000, 15000, n).round(0),
        "region": rng.choice(["east", "west", "north", "south"], n),
        "plan_type": rng.choice(["basic", "premium"], n),
    })
    # inject some missing values
    df.loc[rng.choice(n, 10, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "region"] = np.nan

    # target: some signal from age/income/plan, plus noise
    logit = (
        0.03 * (df["age"].fillna(df["age"].median()) - 40)
        + 0.00005 * (df["income"] - 60000)
        + (df["plan_type"] == "premium") * 0.8
        + rng.normal(0, 1, n)
    )
    y = (logit > 0).astype(int)
    return df, y


def main():
    df, y = make_fake_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["age", "income"]
    categorical_features = ["region", "plan_type"]
    # customer_id is intentionally excluded -> dropped by default "remainder"

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
    ])  # remainder="drop" by default -> customer_id is dropped

    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42)),
    ])

    full_pipeline.fit(X_train, y_train)
    print("Test accuracy:", round(full_pipeline.score(X_test, y_test), 4))

    feature_names = full_pipeline.named_steps["preprocessing"].get_feature_names_out()
    print("\nOutput feature names after ColumnTransformer:")
    print(list(feature_names))
    assert not any("customer_id" in name for name in feature_names), "ID column leaked into features!"
    print("\ncustomer_id correctly excluded from model features.")


if __name__ == "__main__":
    main()
