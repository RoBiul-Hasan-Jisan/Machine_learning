# Data Preprocessing

A practical guide to cleaning and preparing data for machine learning — from missing values through building full `Pipeline` + `ColumnTransformer` workflows.

---

## 1. Missing Values

### Why it matters
Most ML models can't handle `NaN` directly (tree-based models like LightGBM/XGBoost are partial exceptions). How you handle missingness can silently bias your model if done carelessly.

### Step 1: Understand *why* data is missing
| Type | Meaning | Example |
|---|---|---|
| **MCAR** (Missing Completely At Random) | Missingness unrelated to any data | Sensor randomly drops a reading |
| **MAR** (Missing At Random) | Missingness related to *other observed* variables | Income missing more often for younger respondents |
| **MNAR** (Missing Not At Random) | Missingness related to the *missing value itself* | High earners refuse to report income |

This matters because naive imputation can introduce bias, especially under MNAR.

### Step 2: Detect it
```python
df.isnull().sum()
df.isnull().mean() * 100   # % missing per column
import missingno as msno
msno.matrix(df)             # visualize missingness patterns
```

### Step 3: Handle it

**Deletion**
```python
df.dropna(axis=0)                 # drop rows with any NaN
df.dropna(axis=1, thresh=len(df)*0.5)  # drop columns >50% missing
```
Use sparingly — only when missingness is small and MCAR, otherwise you lose information/introduce bias.

**Imputation**
| Method | When to use |
|---|---|
| Mean/median | Numeric, roughly symmetric or skewed (median) distributions |
| Mode (most frequent) | Categorical features |
| Constant / "Missing" category | When missingness itself is informative |
| Forward/backward fill | Time series |
| KNN imputation | When features are correlated and you want smarter fills |
| Model-based (e.g., `IterativeImputer`) | Complex relationships between features |

```python
from sklearn.impute import SimpleImputer, KNNImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
knn_imputer = KNNImputer(n_neighbors=5)
```

**Add a missingness indicator** — often more valuable than the imputed value itself:
```python
df["age_was_missing"] = df["age"].isnull().astype(int)
```
This preserves the *signal* that a value was missing (useful under MAR/MNAR) even after you've filled it in.

> **Golden rule:** fit your imputer only on the training set, then `transform` (not `fit_transform`) the validation/test set — otherwise you leak information from test data into training.

---

## 2. Outliers

### Detecting outliers

**Statistical methods**
```python
# Z-score method (assumes roughly normal distribution)
from scipy import stats
z_scores = stats.zscore(df["col"])
outliers = df[abs(z_scores) > 3]

# IQR method (robust, no normality assumption)
Q1, Q3 = df["col"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = df[(df["col"] < lower) | (df["col"] > upper)]
```

**Visual methods:** box plots, scatter plots, histograms.

**Model-based methods** (for multivariate outliers):
- **Isolation Forest** — isolates anomalies via random partitioning
- **Local Outlier Factor (LOF)** — density-based
- **DBSCAN** — flags low-density points as noise

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05)
outlier_flags = iso.fit_predict(df[["col1", "col2"]])  # -1 = outlier
```

### Handling outliers
| Approach | Notes |
|---|---|
| **Remove** | Only if you're confident it's a data error, and the dataset is large enough to absorb the loss |
| **Cap/Winsorize** | Clip values beyond a percentile (e.g., 1st/99th) instead of deleting |
| **Transform** | Log/Box-Cox transforms shrink the influence of extreme values |
| **Keep + use robust methods** | Use median/IQR-based scaling, or models robust to outliers (tree-based models, robust regression) |
| **Treat as a separate signal** | Sometimes outliers *are* the interesting cases (e.g., fraud) — don't remove them! |

```python
# Winsorizing
df["col"] = df["col"].clip(lower=lower, upper=upper)
```

> Always ask *why* an outlier exists before removing it — a data entry error vs. a genuine rare event should be treated very differently.

---

## 3. Encoding Categorical Variables

Models need numbers, not strings. The right encoding depends on cardinality and whether the categories are ordered.

| Encoding | Best for | Notes |
|---|---|---|
| **One-Hot Encoding** | Low-cardinality nominal (unordered) categories | Creates one binary column per category; watch out for the "dummy variable trap" (drop one column for linear models) |
| **Ordinal Encoding** | Ordered categories (e.g., low/medium/high) | Preserves rank; wrong to use on unordered data — implies a false order |
| **Label Encoding** | Target labels themselves, or tree-based models with nominal features | Tree models can handle arbitrary integer codes fine; linear models cannot |
| **Target/Mean Encoding** | High-cardinality categoricals | Replace category with mean of target for that category — powerful but **prone to leakage**, must be done with cross-validation folds |
| **Frequency/Count Encoding** | High-cardinality categoricals | Replace category with its frequency in the dataset |
| **Hashing Encoding** | Very high-cardinality (e.g., millions of IDs) | Fixed-size output via hash function; avoids huge one-hot matrices, at the cost of collisions |
| **Embeddings** | Very high-cardinality, deep learning | Learn a dense vector representation per category |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal = OrdinalEncoder(categories=[["low", "medium", "high"]])
```

**Target encoding with leakage protection:**
```python
from category_encoders import TargetEncoder
# Always fit target encoders inside cross-validation, never on the full dataset upfront
te = TargetEncoder()
```

---

## 4. Standardization

Rescales features to have **mean = 0, standard deviation = 1**.

$$
z = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # never re-fit on test data
```

**When to use:**
- Algorithms sensitive to feature scale/distance: KNN, SVM, K-Means, PCA, gradient-descent-based linear/logistic regression, neural networks.
- Not needed for tree-based models (decision trees, random forests, gradient boosting) — splits are scale-invariant.

**Robust alternative** (less sensitive to outliers, uses median/IQR instead of mean/std):
```python
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
```

---

## 5. Normalization

Two different things share this name — know which one is meant:

**A. Min-Max Normalization** — rescales values to a fixed range, typically [0, 1]:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

```python
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler(feature_range=(0, 1))
```
Sensitive to outliers (a single extreme value compresses everything else). Useful for neural nets, image pixel data, and algorithms needing bounded input.

**B. Vector (Unit-Norm) Normalization** — rescales *each row* so its vector length (norm) = 1:

```python
from sklearn.preprocessing import Normalizer
row_normalizer = Normalizer(norm="l2")  # scales across features, per sample
```
Used in text (TF-IDF vectors), clustering with cosine similarity, and when the *direction* of the feature vector matters more than its magnitude.

**Standardization vs. Normalization — quick distinction:**
| | Standardization | Min-Max Normalization |
|---|---|---|
| Output range | Unbounded (centered at 0) | Fixed range, e.g. [0,1] |
| Sensitive to outliers? | Less | Yes |
| Assumes distribution? | Works best if roughly Gaussian | No assumption |
| Common use | Linear models, SVM, PCA | Neural nets, image data, bounded-input algorithms |

---

## 6. Feature Transformation

Reshaping a feature's distribution or creating new representations to help the model learn better.

### Skew-correcting transforms
```python
import numpy as np
df["log_col"] = np.log1p(df["col"])          # log(1+x), handles right-skew, zeros safe

from scipy.stats import boxcox
df["boxcox_col"], lam = boxcox(df["col"] + 1)  # requires positive values

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")    # handles negative values too, unlike box-cox
```

### Binning / Discretization
Turns continuous variables into categorical buckets — useful for capturing non-linear effects in linear models, or reducing outlier influence.
```python
from sklearn.preprocessing import KBinsDiscretizer
kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
```

### Polynomial / Interaction Features
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False)
```
Lets linear models capture non-linear relationships (e.g., `x1 * x2`, `x1²`) — but can explode dimensionality quickly.

### Dimensionality Reduction (as a transform step)
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # keep 95% of variance
```
Reduces correlated/high-dimensional features into fewer, uncorrelated components. Always standardize before PCA — it's variance-based and scale-sensitive.

### Date/Time Feature Extraction
```python
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
```

---

## 7. `ColumnTransformer`

Applies **different preprocessing to different columns** in one object — the standard way to handle mixed numeric + categorical data.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_features = ["age", "income"]
categorical_features = ["gender", "city"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
```

**Key points:**
- Columns not listed are dropped by default (`remainder="drop"`) — set `remainder="passthrough"` to keep them untouched.
- Each sub-pipeline is fit *only* on training data, avoiding leakage.
- Works seamlessly inside a full `Pipeline` (see below), and inside `GridSearchCV`/`cross_val_score`.

---

## 8. `Pipeline`

Chains preprocessing + model into a **single object**, ensuring the exact same transformations happen at train and inference time, and preventing data leakage during cross-validation.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),   # the ColumnTransformer from above
    ("classifier", LogisticRegression(max_iter=1000))
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
```

### Why use it (not just call transforms manually)
1. **No leakage** — `fit_transform` only happens on training folds during cross-validation; `.transform` is applied to validation/test automatically.
2. **Reproducibility** — one object encapsulates the entire process, easy to save/load with `joblib`.
3. **Cleaner hyperparameter tuning** — tune preprocessing and model hyperparameters together:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "classifier__C": [0.1, 1, 10]
}

grid = GridSearchCV(full_pipeline, param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)
```

4. **Cross-validation correctness:**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring="roc_auc")
```
Passing the *whole pipeline* (not pre-transformed data) into `cross_val_score` guarantees imputers/scalers/encoders are refit per fold — the correct way to avoid leakage.

### Persisting a pipeline
```python
import joblib
joblib.dump(full_pipeline, "model_pipeline.pkl")
loaded = joblib.load("model_pipeline.pkl")
loaded.predict(new_data)   # applies identical preprocessing automatically
```

---

## Full Worked Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=300, random_state=42))
])

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

---

## Quick Reference: Preprocessing Checklist

| Step | Ask yourself |
|---|---|
| Missing values | Why is it missing? Impute, flag, or drop? |
| Outliers | Data error or genuine signal? Cap, transform, or keep? |
| Categorical encoding | Ordered or unordered? High or low cardinality? |
| Scaling | Does my model care about feature scale (distance/gradient-based)? |
| Transformations | Is the feature skewed? Would interactions/binning help? |
| Leakage check | Am I fitting anything (imputer, scaler, encoder) on data outside the training fold? |
| Pipeline | Is preprocessing + model bundled into one object for CV/deployment? |

---

## Suggested Learning Path
1. Load a messy real dataset (e.g., Titanic) and manually inspect missingness and outliers.
2. Build a `ColumnTransformer` by hand for numeric + categorical columns.
3. Wrap it in a `Pipeline` with a classifier and run `cross_val_score`.
4. Compare model performance with and without scaling for a distance-based model like KNN.
5. Try target encoding vs. one-hot encoding on a high-cardinality column and compare leakage risk.
6. Save and reload a fitted pipeline with `joblib`, and confirm predictions match.
