

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
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
    df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "region"] = np.nan

    logit = (
        0.03 * (df["age"].fillna(df["age"].median()) - 40)
        + 0.00004 * (df["income"] - 60000)
        + 0.02 * (df["tenure_months"] - 36)
        + (df["plan_type"] == "premium") * 0.8
        + rng.normal(0, 1.2, n)
    )
    y = (logit > 0).astype(int)
    return df, y


def build_feature_pipeline(k_best=6):
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
        ("select", SelectKBest(score_func=f_classif, k=k_best)),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def main():
    df, y = make_fake_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_feature_pipeline(k_best=6)
    pipe.fit(X_train, y_train)
    print("Test accuracy:", round(pipe.score(X_test, y_test), 4))

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print("Safe (in-pipeline) CV accuracy: mean =", round(cv_scores.mean(), 4),
          "std =", round(cv_scores.std(), 4))

    # --- Leaky comparison: select features using the FULL dataset's correlation
    # with the target before splitting, then evaluate with CV on the training set ---
    from sklearn.compose import ColumnTransformer as CT
    numeric_features = ["age", "income", "tenure_months"]
    categorical_features = ["region", "plan_type"]
    pre = CT([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])
    X_full_transformed = pre.fit_transform(df, y)          # fit on FULL data, including test rows
    selector_leaky = SelectKBest(score_func=f_classif, k=6).fit(X_full_transformed, y)  # sees all labels
    selected_mask = selector_leaky.get_support()

    X_train_t = pre.transform(X_train)
    X_train_selected = X_train_t[:, selected_mask]
    leaky_model = LogisticRegression(max_iter=1000)
    leaky_cv = cross_val_score(leaky_model, X_train_selected, y_train, cv=5)

    print("\nLeaky (full-dataset) feature selection CV accuracy: mean =",
          round(leaky_cv.mean(), 4))
    print("Note: leaky selection used test-set labels to choose features before")
    print("evaluation, which can inflate the apparent CV score on harder datasets.")


if __name__ == "__main__":
    main()
