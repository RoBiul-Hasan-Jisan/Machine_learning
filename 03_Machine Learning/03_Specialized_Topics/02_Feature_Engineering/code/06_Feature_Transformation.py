"""

Reshape a feature's distribution -- most useful for models that assume
linearity or roughly-normal inputs (linear/logistic regression) and for
reducing the influence of extreme values on tree splits or distance
metrics.
"""

import numpy as np


# Log / sqrt transforms

def log_transform(x, offset=1.0):
    """
    log(x + offset). Compresses large values, expands small ones --
    the standard fix for right-skewed data (revenue, prices, counts).
    Requires x + offset > 0; the default offset=1 handles x=0 safely
    (log1p is more numerically stable than log(x+1) for small x).
    """
    return np.log1p(x + offset - 1)


def sqrt_transform(x):
    """
    Square root. A milder compression than log -- less aggressive on
    large values, and unlike log it's defined at x=0 (still requires
    x >= 0).
    """
    return np.sqrt(np.clip(x, 0, None))


# Box-Cox (positive data only)

def _boxcox_transform(x, lam):
    if abs(lam) < 1e-8:
        return np.log(x)
    return (np.power(x, lam) - 1) / lam


def _boxcox_log_likelihood(x, lam):
    """
    Log-likelihood of the Box-Cox-transformed data under a normal
    distribution, used to pick the best lambda. This is the standard
    profile log-likelihood formula for Box-Cox.
    """
    n = len(x)
    y = _boxcox_transform(x, lam)
    var = y.var()
    if var <= 0:
        return -np.inf
    return -n / 2 * np.log(var) + (lam - 1) * np.sum(np.log(x))


def box_cox(x, lam=None, lam_grid=None):
    """
    Box-Cox power transform: finds the lambda that makes the transformed
    data closest to normally distributed (by maximizing log-likelihood),
    then applies it. Requires strictly positive x.

        lambda = 1    -> no change (linear)
        lambda = 0    -> log(x)
        lambda = 0.5  -> sqrt-like
        lambda = -1   -> 1/x-like

    Args:
        x: strictly positive 1D array
        lam: if provided, skip the search and use this lambda directly
        lam_grid: candidate lambdas to search over if lam is None
                  (default: -2 to 2 in steps of 0.05)

    Returns:
        (transformed_x, chosen_lambda)
    """
    assert np.all(x > 0), "Box-Cox requires strictly positive values"

    if lam is not None:
        return _boxcox_transform(x, lam), lam

    if lam_grid is None:
        lam_grid = np.arange(-2, 2.01, 0.05)

    best_lam, best_ll = 1.0, -np.inf
    for candidate in lam_grid:
        ll = _boxcox_log_likelihood(x, candidate)
        if ll > best_ll:
            best_lam, best_ll = candidate, ll

    return _boxcox_transform(x, best_lam), best_lam


# Yeo-Johnson (handles zero and negative values)

def _yeojohnson_transform(x, lam):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    pos = x >= 0
    neg = ~pos

    if abs(lam) > 1e-8:
        out[pos] = (np.power(x[pos] + 1, lam) - 1) / lam
    else:
        out[pos] = np.log1p(x[pos])

    if abs(lam - 2) > 1e-8:
        out[neg] = -(np.power(-x[neg] + 1, 2 - lam) - 1) / (2 - lam)
    else:
        out[neg] = -np.log1p(-x[neg])

    return out


def _yeojohnson_log_likelihood(x, lam):
    n = len(x)
    y = _yeojohnson_transform(x, lam)
    var = y.var()
    if var <= 0:
        return -np.inf
    # Jacobian term: sum of sign(x) * log(|x| + 1), weighted by (lam - 1)
    jacobian = np.sum(np.sign(x) * np.log1p(np.abs(x)))
    return -n / 2 * np.log(var) + (lam - 1) * jacobian


def yeo_johnson(x, lam=None, lam_grid=None):
    """
    Yeo-Johnson power transform: same goal as Box-Cox (find the power
    transform that best normalizes the distribution), but defined for
    ALL real values (zero and negative included), via a piecewise
    formula that handles the positive and negative parts separately.

    Returns:
        (transformed_x, chosen_lambda)
    """
    x = np.asarray(x, dtype=float)

    if lam is not None:
        return _yeojohnson_transform(x, lam), lam

    if lam_grid is None:
        lam_grid = np.arange(-2, 2.01, 0.05)

    best_lam, best_ll = 1.0, -np.inf
    for candidate in lam_grid:
        ll = _yeojohnson_log_likelihood(x, candidate)
        if ll > best_ll:
            best_lam, best_ll = candidate, ll

    return _yeojohnson_transform(x, best_lam), best_lam


# Polynomial features

def polynomial_features(X, degree=2, include_interactions=True):
    """
    Expand a feature matrix with polynomial terms: x^2, x^3, ... up to
    `degree`, and (if include_interactions) pairwise products x_i * x_j.
    Lets a LINEAR model capture curvature and interaction effects it
    otherwise couldn't, at the cost of more features (and more
    overfitting risk -- pair with feature selection, Section 07).

    Args:
        X: (n_samples, n_features) array
        degree: highest power to include for each individual feature
        include_interactions: whether to add pairwise products (degree-2 only)

    Returns:
        (expanded_X, feature_names)
    """
    n_samples, n_features = X.shape
    columns = [X]
    names = [f"x{i}" for i in range(n_features)]

    for d in range(2, degree + 1):
        for i in range(n_features):
            columns.append((X[:, i] ** d).reshape(-1, 1))
            names.append(f"x{i}^{d}")

    if include_interactions and n_features > 1:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                columns.append((X[:, i] * X[:, j]).reshape(-1, 1))
                names.append(f"x{i}*x{j}")

    return np.hstack(columns), names




def _demo():
  
    print("  FEATURE TRANSFORMATION DEMO")
   

    rng = np.random.RandomState(0)
    # right-skewed data, like income or revenue
    skewed = rng.lognormal(mean=3, sigma=1, size=500)
    print(f"\nRight-skewed data: skewness before = {_skewness(skewed):.3f}")

    log_x = log_transform(skewed, offset=1.0)
    print(f"After log transform:   skewness = {_skewness(log_x):.3f}")

    sqrt_x = sqrt_transform(skewed)
    print(f"After sqrt transform:  skewness = {_skewness(sqrt_x):.3f}")

    bc_x, bc_lambda = box_cox(skewed)
    print(f"After Box-Cox:         skewness = {_skewness(bc_x):.3f}  (chosen lambda={bc_lambda:.2f})")

    # Yeo-Johnson on data that includes negatives (Box-Cox can't handle this)
    signed = skewed - skewed.mean()  # now has negative values
    yj_x, yj_lambda = yeo_johnson(signed)
    print(f"\nData with negative values (Box-Cox would fail here):")
    print(f"  Before Yeo-Johnson: skewness = {_skewness(signed):.3f}")
    print(f"  After Yeo-Johnson:  skewness = {_skewness(yj_x):.3f}  (chosen lambda={yj_lambda:.2f})")

    X = np.array([[2.0, 3.0], [1.0, 5.0], [4.0, 2.0]])
    expanded, names = polynomial_features(X, degree=2, include_interactions=True)
    print(f"\nPolynomial features (degree=2, with interactions):")
    print(f"  Original shape: {X.shape} -> Expanded shape: {expanded.shape}")
    print(f"  Feature names: {names}")
    print(f"  First row: {X[0]} -> {np.round(expanded[0], 2)}")


def _skewness(x):
    """Simple from-scratch skewness (third standardized moment)."""
    mean, std = x.mean(), x.std()
    return np.mean(((x - mean) / std) ** 3)


if __name__ == "__main__":
    _demo()
