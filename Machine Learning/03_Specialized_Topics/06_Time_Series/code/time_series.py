"""
Time Series Fundamentals toolkit: stationarity
checks, autocorrelation, lag/rolling features, walk-forward validation,
a simple AR model, evaluation metrics, and a runnable demo -- all in one
place. Sections are separated by banner comments below.


"""

import numpy as np





def make_synthetic_series(n=500, seed=42):
    """
    Trend + monthly-ish seasonality + noise. Good for demonstrating
    stationarity checks and differencing (has a clear linear trend).
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)

    trend = 0.05 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 30)
    noise = rng.normal(0, 2, n)
    series = 50 + trend + seasonality + noise

    return series


def make_seasonal_series(n=365, period=7, seed=42):
    """
    Trend + weekly seasonality + monthly seasonality + noise. Good for
    demonstrating ACF spikes at lag 7 (and its harmonics) and for lag
    selection experiments.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)

    trend = 0.02 * t
    weekly = 5 * np.sin(2 * np.pi * t / period)
    monthly = 3 * np.sin(2 * np.pi * t / 30)
    noise = rng.normal(0, 1.5, n)
    series = 100 + trend + weekly + monthly + noise

    return series



# STATIONARITY


def difference(series, order=1):
    """
    Apply `order` rounds of first-differencing: diff[t] = value[t] - value[t-1].

    Each round removes one order of polynomial trend:
      - a linear trend needs 1 round to become flat
      - a quadratic trend needs 2 rounds

    Example:
        [100, 102, 106, 112, 120]   (quadratic trend)
        -> [2, 4, 6, 8]             (order=1, still trending -- linear now)
        -> [2, 2, 2]                (order=2, constant -- stationary)

    Note the output shrinks by `order` elements each round, since you lose
    one point per differencing step (there's no "value before the first").
    """
    result = series.copy()
    for _ in range(order):
        result = result[1:] - result[:-1]
    return result


def check_stationarity(series, window=50):
    """
    Two complementary stationarity checks, bundled together:

    1. Rolling mean / rolling std (visual/numeric):
       Computed over a sliding window. If these drift as the window moves
       through the series, the series is non-stationary. Good for plotting
       and building intuition.

    2. First-half vs second-half comparison (quick numeric flag):
       Split the series in half and compare summary stats.
         - mean_shift: how far apart are the two halves' means, in units
           of the whole series' standard deviation?
         - var_ratio: how much bigger is the larger half's variance than
           the smaller half's?
       Flag as non-stationary if the mean shifted by more than half a
       standard deviation, OR the variance changed by more than 2x.
       These thresholds are heuristic, not a formal statistical test --
       see the note below on the ADF test for a rigorous alternative.

    Returns:
        rolling_mean, rolling_std, is_stationary (bool)

    For a formal hypothesis test instead of this heuristic, use the
    Augmented Dickey-Fuller (ADF) test from statsmodels:

        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series)
        p_value = result[1]
        is_stationary = p_value < 0.05   # reject "non-stationary" null hypothesis

    We don't reimplement ADF here because it depends on precomputed
    asymptotic distribution tables for its critical values -- not
    something worth hand-rolling. This module's checks are the fast,
    dependency-free gut-check to run before reaching for statsmodels.
    """
    n = len(series)
    rolling_mean = np.zeros(n)
    rolling_std = np.zeros(n)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = series[start:i + 1]
        rolling_mean[i] = segment.mean()
        rolling_std[i] = segment.std() if len(segment) > 1 else 0.0

    first_half_mean = series[:n // 2].mean()
    second_half_mean = series[n // 2:].mean()
    first_half_var = series[:n // 2].var()
    second_half_var = series[n // 2:].var()

    mean_shift = abs(first_half_mean - second_half_mean)
    var_ratio = max(first_half_var, second_half_var) / max(
        min(first_half_var, second_half_var), 1e-10
    )

    is_stationary = mean_shift < 0.5 * series.std() and var_ratio < 2.0

    return rolling_mean, rolling_std, is_stationary



# AUTOCORRELATION


def autocorrelation(series, max_lag=20):
    """
    Compute the autocorrelation function (ACF) for lags 0..max_lag.

    For each lag k, this measures:
        corr(series[t], series[t-k])  averaged across all valid t

    acf[0] is always 1.0 (a series perfectly correlates with itself at
    lag 0). Values near acf[0]'s magnitude at later lags indicate strong
    memory; values near zero indicate the series has "forgotten" that far
    back.

    Returns:
        np.ndarray of length max_lag + 1, where acf[k] is the
        autocorrelation at lag k.
    """
    n = len(series)
    mean = series.mean()
    var = series.var()
    acf = np.zeros(max_lag + 1)

    for k in range(max_lag + 1):
        if k >= n:
            break
        cov = np.mean((series[:n - k] - mean) * (series[k:] - mean)) if k < n else 0
        acf[k] = cov / var if var > 0 else 0.0

    return acf


def significance_threshold(n_samples):
    """
    Approximate 95% significance threshold for ACF values under the null
    hypothesis of no autocorrelation: +/- 1.96 / sqrt(n).

    Any |acf[k]| above this threshold is a lag worth paying attention to
    (statistically distinguishable from pure noise at ~95% confidence).
    """
    return 1.96 / np.sqrt(n_samples)



# FEATURES


def make_lag_features(series, n_lags):
    """
    Build a lag-feature matrix from a 1D series.

    For n_lags=3 and series [10, 12, 14, 13, 15, 11]:
        row for target=13 (index 3): [lag1=14, lag2=12, lag3=10]
        row for target=15 (index 4): [lag1=13, lag2=14, lag3=12]
        row for target=11 (index 5): [lag1=15, lag2=13, lag3=14]
    (the first n_lags rows are dropped since they don't have n_lags of
    history behind them)

    Returns:
        X: (n_valid, n_lags) feature matrix, column j = lag (j+1)
        y: (n_valid,) target vector, aligned so y[i] corresponds to X[i]
    """
    n = len(series)
    X = np.full((n, n_lags), np.nan)

    for lag in range(1, n_lags + 1):
        X[lag:, lag - 1] = series[:-lag]

    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    y_valid = series[valid_mask]

    return X_valid, y_valid


def rolling_features(series, windows=(7, 14)):
    """
    Rolling mean / std / min / max over each window size, computed using
    only past values (window ending at t-1, never including t -- same
    leakage rule as lag features applies here).

    These give a model information lag features alone can't capture:
      - rolling mean rising  -> suggests an upward trend right now
      - rolling std rising   -> suggests growing volatility right now
    Tree-based models can exploit this kind of "recent regime" signal in
    ways plain lag values don't expose directly.

    Returns a dict of {feature_name: np.ndarray}, each aligned to `series`
    (values before enough history exists are NaN -- combine with
    make_lag_features' valid_mask logic, or drop NaNs, before fitting).
    """
    n = len(series)
    out = {}
    for w in windows:
        roll_mean = np.full(n, np.nan)
        roll_std = np.full(n, np.nan)
        roll_min = np.full(n, np.nan)
        roll_max = np.full(n, np.nan)
        for i in range(w, n):
            window_vals = series[i - w:i]  # strictly before t=i
            roll_mean[i] = window_vals.mean()
            roll_std[i] = window_vals.std()
            roll_min[i] = window_vals.min()
            roll_max[i] = window_vals.max()
        out[f"roll_mean_{w}"] = roll_mean
        out[f"roll_std_{w}"] = roll_std
        out[f"roll_min_{w}"] = roll_min
        out[f"roll_max_{w}"] = roll_max
    return out



# VALIDATION


def walk_forward_split(n_samples, n_splits=5, min_train=50):
    """
    Yield (train_slice, test_slice) pairs where train always precedes test
    chronologically. This implementation is "expanding window": the
    training set always starts at index 0 and grows by `step` samples
    each fold, while the test set is the next `step` samples after it.

    Example with n_samples=200, n_splits=5, min_train=50:
        step = (200 - 50) // 5 = 30
        fold 1: train=[0:50],   test=[50:80]
        fold 2: train=[0:80],   test=[80:110]
        fold 3: train=[0:110],  test=[110:140]
        fold 4: train=[0:140],  test=[140:170]
        fold 5: train=[0:170],  test=[170:200]

    For a *sliding* window instead (fixed-size training set that also
    slides forward, useful when old data becomes actively misleading due
    to regime changes), replace `slice(0, train_end)` below with
    `slice(train_end - min_train, train_end)`.

    Args:
        n_samples: total number of rows in your feature matrix
        n_splits: how many train/test folds to produce
        min_train: minimum number of samples required before the first
            test fold -- protects against training on too little history

    Yields:
        (train_slice, test_slice) -- Python slice objects for indexing
        into your X/y arrays.
    """
    if n_samples <= min_train:
        return

    step = max(1, (n_samples - min_train) // n_splits)

    for i in range(n_splits):
        train_end = min_train + i * step
        test_start = train_end
        test_end = min(train_end + step, n_samples)

        if test_start >= n_samples:
            break

        yield slice(0, train_end), slice(test_start, test_end)



# MODELS


class SimpleAR:
    """
    AR(n_lags) model: predicts y[t] as a linear combination of
    y[t-1] .. y[t-n_lags], fit by ordinary least squares.
    """

    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit via the normal equations (closed-form OLS), solved with
        np.linalg.lstsq for numerical stability rather than explicitly
        inverting X^T X.
        """
        X_b = np.column_stack([np.ones(len(X)), X])
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.bias = theta[0]
        self.weights = theta[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def fit_series(self, series):
        """Convenience: build lag features internally, then fit."""
        X, y = make_lag_features(series, self.n_lags)
        return self.fit(X, y)

    def forecast(self, last_values, n_steps):
        """
        Multi-step forecast using the RECURSIVE strategy: predict one
        step ahead, append that prediction to the history, then use it
        as an input for the next step. Simple, but errors compound --
        each prediction depends on the previous (possibly wrong) one, so
        accuracy degrades as n_steps grows.

        For the DIRECT strategy instead (train a separate model per
        horizon, no error compounding but less data per model and no
        information sharing across horizons), fit a distinct SimpleAR
        instance per target horizon rather than calling this repeatedly.

        Args:
            last_values: array of at least n_lags most-recent true values
            n_steps: how many steps ahead to forecast

        Returns:
            np.ndarray of length n_steps
        """
        if len(last_values) < self.n_lags:
            raise ValueError(
                f"Need at least {self.n_lags} history points, got {len(last_values)}"
            )
        history = list(last_values[-self.n_lags:])
        predictions = []

        for _ in range(n_steps):
            # features are the most recent n_lags values, most-recent first
            features = np.array(history[-self.n_lags:][::-1]).reshape(1, -1)
            pred = self.predict(features)[0]
            predictions.append(pred)
            history.append(pred)  # feed prediction back in as if it were real

        return np.array(predictions)



# METRICS


def mse(y_true, y_pred):
    """Mean Squared Error -- penalizes large errors disproportionately."""
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """
    Mean Absolute Error -- easiest to interpret in original units:
    "predictions are off by X on average."
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error -- same units as the original series (unlike
    MSE), but still penalizes big misses harder than MAE. Use when large
    errors are disproportionately costly in your application.
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error -- scale-independent, useful for
    comparing accuracy across series with very different magnitudes.
    Undefined (division by zero) wherever y_true == 0, so those points
    are excluded here rather than raising.
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def naive_baseline_mse(y_true, lag1_values):
    """
    MSE of the "predict the previous value" baseline. ALWAYS compute this
    alongside any model's score. If your model can't beat it, the most
    likely causes are (in order of frequency): future leakage in a
    feature, an incorrect evaluation split, or the series genuinely has
    no learnable structure beyond persistence/seasonality.
    """
    return mse(y_true, lag1_values)



# DEMO


def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_stationarity():
    print_separator("1. STATIONARITY CHECK")

    series = make_synthetic_series(n=300, seed=42)
    _, _, is_stat = check_stationarity(series)
    print("Original series (trend + seasonality):")
    print(f"  Mean: {series.mean():.2f}, Std: {series.std():.2f}")
    print(f"  Stationary: {is_stat}")

    diff1 = difference(series, order=1)
    _, _, is_stat1 = check_stationarity(diff1)
    print("\nAfter first differencing:")
    print(f"  Mean: {diff1.mean():.4f}, Std: {diff1.std():.2f}")
    print(f"  Stationary: {is_stat1}")


def demo_autocorrelation():
    print_separator("2. AUTOCORRELATION (ACF)")

    series = make_seasonal_series(n=365, period=7, seed=42)
    diff_series = difference(series, order=1)
    acf = autocorrelation(diff_series, max_lag=30)
    threshold = significance_threshold(len(diff_series))

    print("ACF of differenced series (first 15 lags):")
    print(f"{'Lag':>5} {'ACF':>8} {'Significant':>12}")
    print("-" * 28)
    for k in range(15):
        sig = "***" if abs(acf[k]) > threshold else ""
        bar = "#" * int(abs(acf[k]) * 30)
        print(f"{k:>5} {acf[k]:>8.4f} {sig:>4} {bar}")

    print(f"\n95% significance threshold: +/-{threshold:.4f}")
    print("Lags 7 and 14 should show spikes -> weekly seasonality")


def demo_lag_features():
    print_separator("3. LAG FEATURES + AR MODEL")

    series = make_synthetic_series(n=400, seed=42)
    n_lags = 10

    X, y = make_lag_features(series, n_lags)
    print(f"Series length: {len(series)}")
    print(f"Feature matrix: {X.shape} (samples x lag features)")
    print(f"Target vector:  {y.shape}")

    print("\nFirst 3 samples (only first 5 lag cols shown):")
    for i in range(3):
        lags_str = ", ".join(f"{v:.1f}" for v in X[i, :5])
        print(f"  Lags: [{lags_str}, ...] -> Target: {y[i]:.1f}")

    ar = SimpleAR(n_lags=n_lags)
    ar.fit(X, y)

    print(f"\nAR({n_lags}) weights (larger |weight| = more influence):")
    for i, w in enumerate(ar.weights):
        print(f"  Lag {i + 1}: {w:+.4f}")
    print(f"  Bias:  {ar.bias:+.4f}")


def demo_walk_forward():
    print_separator("4. WALK-FORWARD VALIDATION")

    series = make_synthetic_series(n=400, seed=42)
    n_lags = 10
    X, y = make_lag_features(series, n_lags)

    n_splits = 5
    fold_scores = []

    print(f"Walk-forward with {n_splits} splits:")
    print(f"{'Fold':>6} {'Train':>10} {'Test':>10} {'MSE':>10} {'MAE':>10}")
    print("-" * 48)

    for fold, (train_sl, test_sl) in enumerate(
        walk_forward_split(len(X), n_splits=n_splits, min_train=100)
    ):
        X_train, y_train = X[train_sl], y[train_sl]
        X_test, y_test = X[test_sl], y[test_sl]

        ar = SimpleAR(n_lags=n_lags)
        ar.fit(X_train, y_train)
        y_pred = ar.predict(X_test)

        fold_mse = mse(y_test, y_pred)
        fold_mae = mae(y_test, y_pred)
        fold_scores.append(fold_mse)

        print(
            f"{fold + 1:>6} {X_train.shape[0]:>10} {X_test.shape[0]:>10} "
            f"{fold_mse:>10.4f} {fold_mae:>10.4f}"
        )

    print(f"\nMean MSE: {np.mean(fold_scores):.4f}")
    print(f"Std MSE:  {np.std(fold_scores):.4f}")


def demo_random_vs_walk_forward():
    print_separator("5. RANDOM SPLIT vs WALK-FORWARD (the leakage gap)")

    series = make_synthetic_series(n=500, seed=42)
    n_lags = 10
    X, y = make_lag_features(series, n_lags)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    ar_random = SimpleAR(n_lags=n_lags)
    ar_random.fit(X[train_idx], y[train_idx])
    random_mse = mse(y[test_idx], ar_random.predict(X[test_idx]))

    wf_scores = []
    for train_sl, test_sl in walk_forward_split(len(X), n_splits=5, min_train=100):
        ar_wf = SimpleAR(n_lags=n_lags)
        ar_wf.fit(X[train_sl], y[train_sl])
        y_pred = ar_wf.predict(X[test_sl])
        wf_scores.append(mse(y[test_sl], y_pred))

    wf_mse = np.mean(wf_scores)

    print(f"Random 80/20 split MSE:  {random_mse:.4f}")
    print(f"Walk-forward mean MSE:   {wf_mse:.4f}")
    print(f"Ratio (random/wf):       {random_mse / wf_mse:.4f}")
    print()
    if random_mse < wf_mse:
        print("Random split gives lower MSE -- this is the optimistic bias")
        print("from future leakage. The walk-forward score is the honest")
        print("estimate of production performance.")
    else:
        print("Walk-forward gives similar or lower MSE here -- this series")
        print("may be stationary enough that leakage isn't a major factor.")


def demo_lag_comparison():
    print_separator("6. LAG COUNT COMPARISON")

    series = make_seasonal_series(n=365, period=7, seed=42)

    print(f"{'n_lags':>8} {'Mean MSE':>12} {'Mean MAE':>12}")
    print("-" * 34)

    for n_lags in [1, 3, 5, 7, 10, 14, 21, 30]:
        X, y = make_lag_features(series, n_lags)

        scores_mse, scores_mae = [], []
        for train_sl, test_sl in walk_forward_split(
            len(X), n_splits=5, min_train=max(60, n_lags + 20)
        ):
            ar = SimpleAR(n_lags=n_lags)
            ar.fit(X[train_sl], y[train_sl])
            y_pred = ar.predict(X[test_sl])
            scores_mse.append(mse(y[test_sl], y_pred))
            scores_mae.append(mae(y[test_sl], y_pred))

        if scores_mse:
            print(f"{n_lags:>8} {np.mean(scores_mse):>12.4f} {np.mean(scores_mae):>12.4f}")


def demo_forecasting():
    print_separator("7. MULTI-STEP FORECASTING (recursive strategy)")

    series = make_synthetic_series(n=300, seed=42)
    train_series = series[:250]
    true_future = series[250:270]

    n_lags = 10
    ar = SimpleAR(n_lags=n_lags)
    X, y = make_lag_features(train_series, n_lags)
    ar.fit(X, y)

    forecast = ar.forecast(train_series, n_steps=20)

    print(f"Training on {len(train_series)} points, forecasting {len(true_future)} steps ahead")
    print()
    print(f"{'Step':>6} {'True':>10} {'Predicted':>10} {'Error':>10}")
    print("-" * 38)

    for i in range(len(true_future)):
        error = true_future[i] - forecast[i]
        print(f"{i + 1:>6} {true_future[i]:>10.2f} {forecast[i]:>10.2f} {error:>+10.2f}")

    print(f"\nForecast MSE:  {mse(true_future, forecast):.4f}")
    print(f"Forecast MAE:  {mae(true_future, forecast):.4f}")
    print(f"Forecast MAPE: {mape(true_future, forecast):.2f}%")
    print("\nNote: recursive forecasting means errors compound -- watch how")
    print("the error tends to grow across the 20 steps.")


if __name__ == "__main__":
    demo_stationarity()
    demo_autocorrelation()
    demo_lag_features()
    demo_walk_forward()
    demo_random_vs_walk_forward()
    demo_lag_comparison()
    demo_forecasting()
