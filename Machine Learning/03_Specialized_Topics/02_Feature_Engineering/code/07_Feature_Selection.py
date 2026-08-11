"""

Six ways to decide which features are worth keeping, from cheapest
(variance threshold) to most expensive (SHAP-based). Filters (variance,
correlation, SelectKBest) never look at a model. Wrappers (RFE) retrain
a model repeatedly. Embedded methods (feature importance, SHAP) reuse a
model you were probably training anyway.

A from-scratch linear regression (via normal equations) is used as the
underlying model for RFE / importance / SHAP-approximation, since it's
simple, fast, and its coefficients are directly interpretable as
importances -- the same role a tree's split-gain plays in a gradient
boosting model.
"""

import numpy as np



# Variance Threshold


def variance_threshold(X, threshold=0.0):
    """
    Drop columns whose variance is <= `threshold`. A near-constant
    column carries almost no information regardless of the target --
    the cheapest possible filter, and a good first pass before anything
    else.

    Returns: (keep_mask, variances)
    """
    variances = X.var(axis=0)
    keep_mask = variances > threshold
    return keep_mask, variances



# Correlation Analysis (drop redundant features)


def correlation_filter(X, threshold=0.9):
    """
    For each pair of features with |correlation| above `threshold`, drop
    the SECOND one (arbitrary but deterministic tie-break) -- they carry
    largely redundant information, so keeping both adds little signal
    while adding multicollinearity risk for linear models.

    Returns: (keep_mask, corr_matrix)
    """
    n_features = X.shape[1]
    corr = np.corrcoef(X, rowvar=False)
    keep_mask = np.ones(n_features, dtype=bool)

    for i in range(n_features):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n_features):
            if keep_mask[j] and abs(corr[i, j]) > threshold:
                keep_mask[j] = False

    return keep_mask, corr



# SelectKBest (via F-statistic, ANOVA-style)


def f_statistic_scores(X, y):
    """
    Compute an F-statistic-like score for each feature independently:
    how much of the feature's variance is "explained" by grouping on
    the target, versus left as noise. For a continuous target this
    reduces to (essentially) the squared correlation, rescaled -- for a
    classification target it compares between-class variance to
    within-class variance (true one-way ANOVA F-test).

    Higher score = feature's values differ more across target values,
    i.e. more informative for separating/predicting the target.
    """
    y = np.asarray(y)
    is_classification = len(np.unique(y)) <= max(20, len(y) // 20) and np.all(y == y.astype(int))

    n_features = X.shape[1]
    scores = np.zeros(n_features)

    if is_classification:
        classes = np.unique(y)
        overall_mean = X.mean(axis=0)
        for f in range(n_features):
            col = X[:, f]
            ss_between, ss_within = 0.0, 0.0
            for c in classes:
                group = col[y == c]
                ss_between += len(group) * (group.mean() - overall_mean[f]) ** 2
                ss_within += ((group - group.mean()) ** 2).sum()
            df_between, df_within = len(classes) - 1, len(y) - len(classes)
            if ss_within <= 0 or df_between <= 0:
                scores[f] = 0.0
            else:
                scores[f] = (ss_between / df_between) / (ss_within / df_within)
    else:
        # continuous target: score = R^2 of a single-feature regression,
        # converted to an F-like statistic
        for f in range(n_features):
            if X[:, f].std() == 0:  # constant column: no relationship to score
                scores[f] = 0.0
                continue
            corr = np.corrcoef(X[:, f], y)[0, 1]
            r2 = corr ** 2 if np.isfinite(corr) else 0.0
            n = len(y)
            scores[f] = (r2 / 1) / ((1 - r2) / (n - 2)) if r2 < 1 else np.inf

    return scores


def select_k_best(X, y, k):
    """Keep the top-k features by f_statistic_scores. Returns (keep_mask, scores)."""
    scores = f_statistic_scores(X, y)
    top_k_idx = np.argsort(-scores)[:k]
    keep_mask = np.zeros(X.shape[1], dtype=bool)
    keep_mask[top_k_idx] = True
    return keep_mask, scores



# A minimal linear model, used by RFE / importance / SHAP below


class _LinearRegression:
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X_b = np.column_stack([np.ones(len(X)), X])
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_



# Recursive Feature Elimination


def recursive_feature_elimination(X, y, n_features_to_select):
    """
    Wrapper method: fit a linear model, drop the feature with the
    smallest |coefficient| (least influence on predictions), refit, and
    repeat until only `n_features_to_select` remain. More expensive than
    a filter (retrains every iteration) but accounts for how features
    interact with each other through the model, not just their
    individual relationship with the target.

    Returns: (keep_mask, elimination_order) where elimination_order
    lists feature indices in the order they were dropped (first dropped
    = least important overall).
    """
    n_features = X.shape[1]
    remaining = list(range(n_features))
    elimination_order = []

    while len(remaining) > n_features_to_select:
        model = _LinearRegression().fit(X[:, remaining], y)
        # coefficients aren't directly comparable unless features are on
        # similar scales -- assume X is already scaled (see 05_Feature_Scaling)
        least_important_local_idx = np.argmin(np.abs(model.coef_))
        dropped_feature = remaining.pop(least_important_local_idx)
        elimination_order.append(dropped_feature)

    keep_mask = np.zeros(n_features, dtype=bool)
    keep_mask[remaining] = True
    return keep_mask, elimination_order



# Feature Importance (coefficient magnitude, standardized)


def feature_importance(X, y):
    """
    Fit one linear model on ALL features and rank by |standardized
    coefficient|. This is the "embedded" method -- reuses a model you
    were likely training anyway, unlike RFE which retrains repeatedly.
    Requires X to already be scaled (see 05_Feature_Scaling) so
    coefficients are comparable across features.
    """
    model = _LinearRegression().fit(X, y)
    importance = np.abs(model.coef_)
    return importance / importance.sum() if importance.sum() > 0 else importance



# SHAP-based Selection (simplified from-scratch approximation)


def shap_approx_importance(X, y, n_samples=200, random_state=42):
    """
    A simplified, from-scratch approximation of mean |SHAP value| per
    feature, for a linear model. For a LINEAR model, the exact Shapley
    value of feature i for a given row is:

        phi_i = coef_i * (x_i - mean(x_i))

    (This is a known closed-form special case of SHAP for linear models
    -- full SHAP's general-purpose power comes from handling non-linear
    models via sampling coalitions of features, which is what makes it
    expensive; for a linear model the exact value has this direct
    formula, so no sampling is actually needed here.)

    We average |phi_i| across a sample of rows to get a per-feature
    importance ranking that (unlike raw coefficient magnitude) is in
    the same units as the target and accounts for each feature's actual
    spread in the data, not just its coefficient.

    Returns: importance scores per feature (higher = more influential
    on average prediction).
    """
    rng = np.random.RandomState(random_state)
    model = _LinearRegression().fit(X, y)

    n = len(X)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    X_sample = X[idx]

    feature_means = X.mean(axis=0)
    phi = model.coef_ * (X_sample - feature_means)  # (n_samples, n_features)

    return np.abs(phi).mean(axis=0)





def _demo():
    
    print("  FEATURE SELECTION DEMO")
   

    rng = np.random.RandomState(0)
    n = 300
    x1 = rng.normal(0, 1, n)             # genuinely predictive
    x2 = x1 * 2 + rng.normal(0, 0.1, n)  # redundant (highly correlated with x1)
    x3 = rng.normal(0, 1, n)             # irrelevant noise
    x4 = np.full(n, 5.0)                 # constant, zero variance
    y = 3 * x1 + rng.normal(0, 0.5, n)   # target depends only on x1

    X = np.column_stack([x1, x2, x3, x4])
    names = ["x1(predictive)", "x2(redundant)", "x3(noise)", "x4(constant)"]

    keep_var, variances = variance_threshold(X, threshold=1e-6)
    print("\nVariance Threshold:")
    for name, v, keep in zip(names, variances, keep_var):
        print(f"  {name}: variance={v:.4f}  keep={keep}")

    keep_corr, corr = correlation_filter(X[:, :3], threshold=0.9)
    print("\nCorrelation Filter (on x1,x2,x3; x2 is redundant with x1):")
    for name, keep in zip(names[:3], keep_corr):
        print(f"  {name}: keep={keep}")

    keep_kbest, scores = select_k_best(X, y, k=2)
    print("\nSelectKBest (k=2, F-statistic scores):")
    for name, s, keep in zip(names, scores, keep_kbest):
        print(f"  {name}: score={s:.2f}  keep={keep}")

    keep_rfe, elim_order = recursive_feature_elimination(X, y, n_features_to_select=2)
    print("\nRFE (target: keep 2 features):")
    print(f"  Elimination order (dropped first -> last): "
          f"{[names[i] for i in elim_order]}")
    print(f"  Final kept: {[names[i] for i, k in enumerate(keep_rfe) if k]}")

    importance = feature_importance(X, y)
    print("\nFeature Importance (standardized |coefficient|, normalized):")
    for name, imp in zip(names, importance):
        print(f"  {name}: {imp:.3f}")

    shap_imp = shap_approx_importance(X, y)
    print("\nSHAP-approx importance (mean |contribution to prediction|):")
    for name, imp in zip(names, shap_imp):
        print(f"  {name}: {imp:.3f}")


if __name__ == "__main__":
    _demo()
