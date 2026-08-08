# Feature Engineering

The art and science of turning raw data into features that make models perform better. Often the single highest-leverage step in a machine learning project — better features regularly beat fancier models.

---

## 1. Feature Creation

The process of deriving new, more informative variables from existing raw data.

### General approaches

**Aggregations** — summarize groups of related rows into a single feature.
```python
# e.g., customer-level features from a transactions table
agg = transactions.groupby("customer_id").agg(
    total_spend=("amount", "sum"),
    avg_spend=("amount", "mean"),
    txn_count=("amount", "count"),
    max_spend=("amount", "max"),
    spend_std=("amount", "std")
).reset_index()
```

**Ratios and rates** — often more informative than raw counts.
```python
df["conversion_rate"] = df["purchases"] / df["visits"]
df["debt_to_income"] = df["debt"] / df["income"]
```

**Flags/indicators** — binary features that capture a condition.
```python
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
df["has_missing_phone"] = df["phone"].isnull().astype(int)
```

**Counts** — number of related items.
```python
df["num_products_owned"] = df[["has_card", "has_loan", "has_savings"]].sum(axis=1)
```

**Text-derived features**
```python
df["title_length"] = df["title"].str.len()
df["word_count"] = df["title"].str.split().str.len()
df["has_exclamation"] = df["title"].str.contains("!").astype(int)
```

### Guiding principle
Good feature creation usually comes from **domain understanding**, not blind automation — ask "what would a human expert look at to make this decision?" and try to encode that signal numerically.

---

## 2. Feature Selection

Reducing the feature set to the most useful subset — improves generalization, reduces overfitting, speeds up training, and improves interpretability.

### A. Filter Methods
Fast, model-agnostic — rank features by a statistical relationship with the target, independent of any specific model.

| Method | Use case |
|---|---|
| **Correlation with target** | Numeric features + numeric/binary target |
| **Chi-squared test** | Categorical features + categorical target |
| **ANOVA F-test** | Numeric features + categorical target |
| **Mutual information** | Captures non-linear relationships, works for mixed types |
| **Variance threshold** | Drop near-constant features (little/no signal) |

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

vt = VarianceThreshold(threshold=0.01)
X_reduced = vt.fit_transform(X_train)
```

### B. Wrapper Methods
Use the actual model's performance to evaluate feature subsets — more accurate, more expensive.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
```
- **Forward selection** — start empty, add the best feature each round.
- **Backward elimination** — start with all features, remove the worst each round.
- **RFE (Recursive Feature Elimination)** — repeatedly fit the model, drop the least important feature.

### C. Embedded Methods
Feature selection happens *as part of* model training.

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X_train, y_train)
important = X_train.columns[lasso.coef_ != 0]   # L1 regularization zeroes out weak features

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
```

### D. Multicollinearity Check
Highly correlated features add redundancy and can destabilize linear model coefficients.
```python
corr_matrix = X_train.corr().abs()
# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns
)  # VIF > 5–10 typically signals problematic collinearity
```

### Quick comparison
| Method type | Speed | Accounts for model? | Risk |
|---|---|---|---|
| Filter | Fast | No | May miss feature interactions |
| Wrapper | Slow | Yes | Expensive, can overfit to validation set |
| Embedded | Medium | Yes | Tied to specific model type |

---

## 3. Numerical Transformations

Reshaping numeric feature distributions to help models learn more effectively.

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer

df["log_income"] = np.log1p(df["income"])          # fix right-skew
df["sqrt_area"] = np.sqrt(df["area"])                # milder skew correction

pt = PowerTransformer(method="yeo-johnson")           # handles negative & zero values
df["yeo_col"] = pt.fit_transform(df[["col"]])

kb = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
df["income_bucket"] = kb.fit_transform(df[["income"]])
```

**When each helps:**
| Transform | Fixes |
|---|---|
| Log / log1p | Right-skewed data, multiplicative relationships |
| Square root | Mild right-skew, count data |
| Box-Cox / Yeo-Johnson | Automatically finds the best power transform toward normality |
| Binning | Captures threshold effects, reduces outlier sensitivity, helps linear models see non-linear patterns |
| Clipping/Winsorizing | Bounds extreme values without dropping rows |

---

## 4. Categorical Features

Beyond basic encoding (covered in preprocessing), feature *engineering* for categoricals focuses on extracting more signal from them.

**Grouping rare categories**
```python
freq = df["city"].value_counts(normalize=True)
rare = freq[freq < 0.01].index
df["city_grouped"] = df["city"].replace(rare, "Other")
```
Prevents overfitting to categories seen only a handful of times, and keeps one-hot dimensionality manageable.

**Combining categories**
```python
df["region_city"] = df["region"] + "_" + df["city"]
```

**Target/frequency encoding** (for high cardinality) — see preprocessing guide; always fit within CV folds to avoid leakage.

**Extracting structure from category strings**
```python
df["email_domain"] = df["email"].str.split("@").str[1]
df["zip_prefix"] = df["zipcode"].str[:3]
```

---

## 5. Date/Time Features

Raw timestamps are almost never directly useful — decompose them.

```python
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["quarter"] = df["date"].dt.quarter
df["week_of_year"] = df["date"].dt.isocalendar().week
df["hour"] = df["date"].dt.hour
```

**Cyclical encoding** — raw month/hour values imply a false "distance" (Dec=12 seems far from Jan=1). Encode cyclically instead:
```python
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

**Time-since / time-until features**
```python
df["days_since_signup"] = (df["current_date"] - df["signup_date"]).dt.days
df["days_until_expiry"] = (df["expiry_date"] - df["current_date"]).dt.days
```

**Holiday/event flags**
```python
import holidays
us_holidays = holidays.US()
df["is_holiday"] = df["date"].isin(us_holidays).astype(int)
```

**Lag and rolling window features** (time series / sequential data):
```python
df["sales_lag_1"] = df.groupby("store_id")["sales"].shift(1)
df["sales_rolling_mean_7"] = df.groupby("store_id")["sales"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
```
> Always `shift(1)` before rolling in these cases — otherwise you leak the current row's own value into its own feature.

---

## 6. Interaction Features

Capture how two (or more) features jointly affect the target, beyond what each does alone.

**Arithmetic interactions**
```python
df["price_per_sqft"] = df["price"] / df["sqft"]
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
```

**Polynomial/automated interactions**
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X[["feature1", "feature2"]])
```
`interaction_only=True` skips squared terms (x1², x2²) and keeps only cross-terms (x1*x2) — often what you want to avoid dimensionality blowup.

**Categorical × Categorical interactions**
```python
df["gender_x_region"] = df["gender"] + "_" + df["region"]
```

**Categorical × Numeric interactions**
```python
# group-relative comparison: how does this row compare to its category's typical value?
df["income_vs_city_avg"] = df["income"] - df.groupby("city")["income"].transform("mean")
```

**When to bother:** Linear models and shallow trees benefit most, since they can't automatically discover interactions on their own. Deep trees/gradient boosting and neural nets can find many interactions implicitly — but explicit domain-informed interactions can still help even there, especially with limited data.

---

## 7. Domain-Based Features

The highest-value, hardest-to-automate category — features built from actual subject-matter knowledge of the problem.

### Examples by domain

**Finance/Credit risk**
```python
df["debt_to_income"] = df["total_debt"] / df["income"]
df["credit_utilization"] = df["balance"] / df["credit_limit"]
df["payment_history_score"] = ...  # domain-specific scoring logic
```

**E-commerce**
```python
df["days_since_last_purchase"] = (today - df["last_purchase_date"]).dt.days
df["avg_order_value"] = df["total_spend"] / df["order_count"]
df["is_repeat_customer"] = (df["order_count"] > 1).astype(int)
```

**Healthcare**
```python
df["bmi_category"] = pd.cut(df["bmi"], bins=[0,18.5,25,30,100],
                             labels=["underweight","normal","overweight","obese"])
```

**Text/NLP**
```python
df["sentiment_score"] = ...      # from a sentiment model
df["readability_score"] = ...    # e.g., Flesch reading ease
```

**Geospatial**
```python
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2*atan2(sqrt(a), sqrt(1-a))

df["distance_to_city_center_km"] = df.apply(
    lambda r: haversine(r["lat"], r["lon"], city_lat, city_lon), axis=1
)
```

### Why domain features matter most
Statistical/automated methods (polynomial features, embeddings, target encoding) find patterns *within the data you already have*. Domain features often inject **external knowledge** the model could never derive from the raw columns alone — e.g., knowing that "debt-to-income ratio" is meaningful because of how lending decisions actually work, not because the data revealed it.

---

## Feature Engineering Workflow (Putting It Together)

1. **Understand the target and business context** — talk to domain experts if possible.
2. **Explore raw data** — distributions, missingness, cardinality, correlations.
3. **Create candidate features** — aggregations, ratios, date parts, domain formulas.
4. **Transform** — fix skew, bin, encode categoricals appropriately.
5. **Generate interactions** — where the model type can't find them on its own.
6. **Select** — filter/wrapper/embedded methods to trim noise and redundancy.
7. **Validate with cross-validation** — confirm new features actually improve out-of-sample performance, not just training score.
8. **Watch for leakage** — never engineer a feature using information that wouldn't be available at prediction time (e.g., using "total lifetime purchases" to predict a customer's *first* purchase).

```python
# Example: checking if a new feature actually helps
from sklearn.model_selection import cross_val_score
baseline_score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
with_new_feature_score = cross_val_score(model, X_train_plus_feature, y_train, cv=5, scoring="roc_auc").mean()
```

---

## Quick Reference: Feature Engineering Checklist

| Category | Key question |
|---|---|
| Creation | What derived quantities would a domain expert calculate by hand? |
| Selection | Which features add real signal vs. noise or redundancy? |
| Numerical transforms | Is the distribution skewed? Would binning reveal thresholds? |
| Categorical | Is cardinality manageable? Are rare categories grouped? |
| Date/time | Have I extracted cyclical, lag, and time-since features? |
| Interactions | Can my model type discover interactions on its own, or do I need to hand it some? |
| Domain | Am I encoding real-world knowledge, not just statistical tricks? |
| Leakage | Would this feature be available at actual prediction time? |

---

## Suggested Learning Path
1. Take a raw dataset (e.g., a retail transactions table) and build 5 aggregation-based features by hand.
2. Apply filter, wrapper, and embedded feature selection to the same dataset and compare which features each keeps.
3. Extract full date/time features (including cyclical encoding) from a timestamp column.
4. Create 2–3 domain-informed features for a dataset in a field you know well, and measure their impact via cross-validation.
5. Deliberately create a leaky feature (e.g., using future information) and observe how it inflates validation scores unrealistically — then fix it.
