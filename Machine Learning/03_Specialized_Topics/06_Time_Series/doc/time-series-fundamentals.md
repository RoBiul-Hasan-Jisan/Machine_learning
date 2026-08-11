# Time Series Fundamentals 

> Past performance does predict future results — if you check for stationarity first.

This is an expanded walkthrough of the original lesson. Each major idea gets its own section with a plain-language explanation, a worked example, and a note on why it matters in practice. The goal is to fill in the "why" behind each technique, not just the "what."

---

## 1. Why Time Series Needs Its Own Toolkit

### 1.1 The i.i.d. assumption, and why it breaks

Standard ML models (linear regression, random forests, most of scikit-learn) are built on an assumption called **i.i.d.** — independent and identically distributed:

- **Independent**: knowing one sample tells you nothing about another.
- **Identically distributed**: every sample comes from the same underlying distribution.

Time series violates both:

- **Not independent** — today's value is correlated with yesterday's. If you know Monday's sales, you have real information about Tuesday's sales.
- **Not identically distributed** — the distribution itself can shift. December retail sales come from a different distribution than March retail sales. A model trained only on spring data has never seen a holiday spike.

### 1.2 Why this matters concretely

Because of these violations:

- **Shuffling data destroys information.** In i.i.d. data, shuffling rows before training changes nothing. In time series, shuffling erases the order that carries the signal.
- **Random cross-validation lies to you.** If you randomly split time-ordered data into train/test, some training rows will sit chronologically *after* some test rows. The model effectively gets to see the future before being tested on the past — a subtle form of cheating that inflates reported accuracy.
- **The "95% accuracy" trap.** A model can score 95% under random cross-validation and 55% under proper time-respecting evaluation. This isn't noise — it's the model having memorized leaked future information rather than learning a real predictive pattern. Section 4 below builds the correct evaluation method (walk-forward validation) to prevent this.

---

## 2. Decomposing a Series: Trend, Seasonality, Residual

Every observed time series can be thought of as the sum (or product) of three components:

| Component | Plain description | Example |
|---|---|---|
| **Trend** | The long-run direction, ignoring short-term wiggles | Revenue growing ~10%/year |
| **Seasonality** | A pattern that repeats at a fixed, known interval | Ice cream sales peak every July |
| **Residual (noise)** | Whatever is left after you remove trend and seasonality | Day-to-day randomness with no explainable cause |

**Why decompose at all?** Two reasons:

1. **Diagnosis.** If you plot the residual after removing trend and seasonality and it still shows a pattern (e.g., a slope, or spikes at a fixed lag), your decomposition missed something — there's more structure to model.
2. **Simplification.** Some models (like exponential smoothing) explicitly model trend and seasonality as separate terms, which is more interpretable than throwing raw values at a black-box model.

A residual that looks like pure white noise (no visible pattern, roughly constant spread) is the sign of a *complete* decomposition — you've extracted everything predictable.

---

## 3. Stationarity — the Concept the Whole Lesson Hinges On

### 3.1 What it means

A series is **stationary** if its statistical fingerprint — mean, variance, and autocorrelation structure — stays constant over time. A stationary series looks statistically "the same" whether you sample it in January or in August.

### 3.2 Why it matters

Most classical forecasting math (ARIMA, and even the intuition behind why linear regression works well on a series) assumes stationarity. If the mean drifts, a model trained on early data has effectively learned the wrong baseline for later data — it will be systematically biased, not just noisy.

### 3.3 How to check it — two complementary methods

**Method A: Rolling statistics (visual/numeric).**
Compute the mean and standard deviation over a sliding window as it moves through the series. If these rolling values trend upward, downward, or change their spread over time, the series is non-stationary.

```python
def check_stationarity(series, window=50):
    rolling_mean = np.array([
        series[max(0, i - window):i].mean()
        for i in range(1, len(series) + 1)
    ])
    rolling_std = np.array([
        series[max(0, i - window):i].std()
        for i in range(1, len(series) + 1)
    ])
    return rolling_mean, rolling_std
```

A quick numeric companion check (used in the reference code): split the series in half, compare the mean and variance of each half. If the means differ by more than half a standard deviation, or the variance ratio between halves exceeds 2x, flag the series as non-stationary.

**Method B: The Augmented Dickey-Fuller (ADF) test (formal statistical test).**
This is the standard hypothesis test for stationarity:

- **Null hypothesis (H0):** the series is non-stationary (it has a "unit root").
- **Decision rule:** if the p-value is below 0.05, you reject H0 and conclude the series is stationary.

The ADF test requires asymptotic distribution tables to compute critical values correctly, so it isn't implemented from scratch here — use `statsmodels.tsa.stattools.adfuller` in practice:

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)
print(f"ADF statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# p < 0.05 -> reject H0 -> series is stationary
```

Use the rolling-statistics method for a fast, visual gut-check during exploration, and the ADF test when you need a defensible, quantified answer (e.g., before choosing an ARIMA order).

### 3.4 Fixing non-stationarity: differencing

Instead of modeling raw values, model the *change* between consecutive values:

```
diff[t] = value[t] - value[t-1]
```

**Worked example.** Original series with a quadratic trend:

```
Original:         [100, 102, 106, 112, 120]
First difference:  [2, 4, 6, 8]        <- still trending (linear now, not quadratic)
Second difference: [2, 2, 2]           <- constant -> stationary
```

Each round of differencing removes one "order" of polynomial trend. A linear trend needs one round; a quadratic trend needs two. In practice, real-world series rarely need more than two rounds — if you find yourself needing a third, something else (like a structural break) is probably going on, and differencing isn't the right tool.

---

## 4. Autocorrelation — Measuring How Much a Series Remembers Itself

### 4.1 ACF (Autocorrelation Function)

ACF measures the correlation between a series and a lagged copy of itself, at every lag k:

- **How far back the series "remembers."** If ACF is essentially zero after lag 5, values more than 5 steps back carry no useful signal.
- **Whether seasonality exists.** A spike in ACF at lag 12 in monthly data is a strong signal of yearly seasonality; a spike at lag 7 in daily data signals weekly seasonality.
- **How many lag features to build.** Use lags up to roughly where ACF becomes negligible (see Section 5).

```python
def autocorrelation(series, max_lag=20):
    n = len(series)
    mean = series.mean()
    var = series.var()
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        cov = np.mean((series[:n-k] - mean) * (series[k:] - mean))
        acf[k] = cov / var if var > 0 else 0
    return acf
```

### 4.2 PACF (Partial Autocorrelation Function) — the piece the original lesson mentioned but didn't fully unpack

ACF can be misleading because correlations propagate indirectly. Suppose today correlates with 3 days ago *only because* today correlates with yesterday, and yesterday correlates with 2 days ago, and so on down the chain. ACF at lag 3 will show a nonzero correlation even though there's no *direct* relationship between today and 3 days ago — it's all mediated through the days in between.

**PACF strips out that indirect chain.** It measures the correlation between today and lag-k *after removing the effect of every shorter lag*. Concretely:

- ACF at lag 3 nonzero + PACF at lag 3 ≈ zero → the lag-3 relationship is fully explained by lags 1 and 2; there's no *direct* 3-day effect.
- ACF and PACF both nonzero at lag 3 → there's a genuine direct relationship 3 steps back, independent of the intermediate days.

**Why this matters practically:** PACF is what tells you the *order* of an autoregressive process. If PACF cuts off sharply to near-zero after lag p, an AR(p) model (using exactly p lags) is likely sufficient — adding more lags would mostly add noise, not signal. ACF alone can't tell you this, because it stays "contaminated" by indirect correlation at every lag.

---

## 5. Lag Features — Turning a Time Series into a Supervised Learning Problem

### 5.1 The core trick

Standard ML models want a feature matrix `X` and target `y`. A time series is just one column of numbers. The bridge is **lag features**: use past values as columns, and the current value as the target.

Series: `[10, 12, 14, 13, 15]` → lag-1 and lag-2 features:

| lag_2 | lag_1 | target |
|-------|-------|--------|
| 10 | 12 | 14 |
| 12 | 14 | 13 |
| 14 | 13 | 15 |

Now it's an ordinary regression problem, solvable by linear regression, random forest, gradient boosting — anything.

### 5.2 Beyond raw lags: engineered features

| Feature type | What it captures |
|---|---|
| Rolling statistics (mean, std, min, max over last k) | Recent trend and volatility |
| Calendar features (day of week, month, is_holiday) | Deterministic seasonal effects |
| Differenced values | Rate of change |
| Expanding statistics (cumulative mean/sum) | Long-run level |
| Ratio features (value / rolling mean) | How unusual the current value is relative to recent history |
| Interaction features (lag_1 × day_of_week) | Weekday-specific momentum effects |

### 5.3 How many lags to use

Don't guess — use ACF/PACF (Section 4). If ACF is significant out to lag 10, include at least 10 lags. If there's weekly seasonality, add lag 7 (and possibly 14) even if consecutive lags 1–6 aren't all significant. More lags = more history but also more parameters to fit, which raises overfitting risk — this is a real trade-off, not just "more is better."

### 5.4 The single most common bug: the target-alignment trap

The target must be the value at time `t`. **Every feature must come from time `t-1` or earlier.** If the value at time `t` accidentally leaks into the feature set (a common off-by-one mistake), you get a "perfect predictor" that is completely useless in production, because at prediction time you don't actually know the value you're trying to predict. Always explicitly audit: for each feature column, what timestamp does this value come from, and is it strictly before the target's timestamp?

---

## 6. Walk-Forward Validation — the Most Important Section

### 6.1 Why random splits fail here

Random k-fold cross-validation assumes samples are interchangeable — true for i.i.d. data, false for time series. If you randomly assign points to train/test, some test points end up *chronologically before* some training points. The model is effectively being graded on a test where it's already seen data from the future relative to that test point. This is why "95% accuracy" can collapse to "55% accuracy" under a fair evaluation — the random split was leaking information.

### 6.2 The correct procedure

1. Train on all data up to time `t`.
2. Predict at `t+1` (or `t+1` through `t+k` for multi-step forecasts).
3. Slide the window forward.
4. Repeat.

Every test fold contains only data that comes strictly after everything in its training fold. No future leakage, and the resulting score is an honest estimate of real-world deployment performance.

### 6.3 Expanding vs. sliding windows

| Strategy | Training set behavior | Use when |
|---|---|---|
| **Expanding window** | Grows every fold, keeps all history | You believe older data is still relevant (stable process) |
| **Sliding window** | Fixed size, old data drops off as new data enters | The underlying process changes over time (regime shifts, concept drift) |

### 6.4 Reference implementation

```python
def walk_forward_split(n_samples, n_splits=5, min_train=50):
    assert min_train < n_samples, "min_train must be less than n_samples"
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)
```

This is an **expanding-window** implementation — `train_end` grows every iteration while `slice(0, train_end)` always starts at 0.

### 6.5 sklearn equivalent

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

# or, more concisely:
scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
print(f"Mean score: {scores.mean():.4f} +/- {scores.std():.4f}")
```

---

## 7. ARIMA — What Each Letter Actually Does

ARIMA(p, d, q) combines three separate ideas into one model:

| Component | Full name | What it does | Chosen using |
|---|---|---|---|
| **AR(p)** | AutoRegressive | Predicts the next value as a weighted combination of the last `p` actual values | PACF (Section 4.2) — look for where PACF cuts off |
| **I(d)** | Integrated | Applies `d` rounds of differencing (Section 3.4) to make the series stationary before modeling | Rolling stats / ADF test — how many differences until stationary |
| **MA(q)** | Moving Average | Predicts the next value as a weighted combination of the last `q` *forecast errors* (not raw values) | ACF — look for where ACF cuts off |

**Concretely:** ARIMA(5, 1, 2) means: difference the series once to make it stationary, then model it as a combination of the last 5 actual (differenced) values plus the last 2 forecast errors.

This lesson doesn't implement ARIMA from scratch — the optimization involved (maximum likelihood estimation over the AR and MA coefficients) is genuinely more involved than a few lines of NumPy. Use `statsmodels` in practice:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(5, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
```

The value of understanding p, d, q conceptually — even without implementing the optimizer — is that it tells you how to *read* an ARIMA model's diagnostics and pick reasonable starting values instead of blindly trusting auto-ARIMA.

---

## 8. Choosing an Approach

| Approach | Best for | Handles seasonality | Handles external features |
|---|---|---|---|
| Lag features + ML (Ridge, GBM) | Tabular data with many external predictors | Via calendar features | Yes |
| ARIMA | Single univariate series, short-term forecasts | SARIMA variant | Limited (ARIMAX) |
| Exponential smoothing | Simple trend + seasonality, few external drivers | Yes (Holt-Winters) | No |
| Prophet | Business forecasting with holidays | Yes (Fourier terms) | Limited |
| Neural nets (LSTM, Transformer) | Long sequences, many related series at once | Learned from data | Yes |

**Default recommendation:** lag features + gradient boosting. It handles external features naturally, doesn't require the stationarity assumptions that ARIMA and linear AR models need, and is straightforward to debug because you can inspect feature importances.

---

## 9. Forecasting Multiple Steps Ahead

| Strategy | How it works | Trade-off |
|---|---|---|
| **Recursive (iterated)** | Predict step 1, feed that prediction back in as if it were real to predict step 2, and so on | Simple, but errors compound — a bad early prediction poisons every later one |
| **Direct** | Train a separate model per horizon (one model for t+1, another for t+5, etc.) | No error accumulation, but each model sees fewer effective examples and models don't share information |
| **Multi-output** | One model predicts all horizons at once | Shares information across horizons, but needs a model architecture that supports multiple outputs |

**Rule of thumb:** recursive for short horizons (1–5 steps), direct for longer horizons where compounding error would dominate.

---

## 10. Evaluation Metrics, With Context

| Metric | Formula (intuition) | When to use |
|---|---|---|
| **MAE** | Average of \|true − predicted\| | Easiest to interpret in original units: "predictions are off by 3.2 degrees on average" |
| **RMSE** | Square root of average squared error | Use when large errors are disproportionately bad — it penalizes big misses harder than MAE |
| **MAPE** | Average of \|error / true value\| × 100 | Scale-independent, good for comparing across series with different magnitudes — but breaks down (undefined) when true values are zero |
| **Naive baseline comparison** | Compare against "predict yesterday's value" or "predict same period last cycle" | Mandatory sanity check — if your model can't beat this, something is broken |

**Why the naive baseline check is non-negotiable:** it's the fastest way to catch bugs. If a sophisticated model loses to "just predict last week's value," the most likely causes are (in order of frequency): future leakage in a feature, an incorrect evaluation split, or the series genuinely has no learnable structure beyond its seasonal pattern.

---

## 11. Common Mistakes, Expanded

| Mistake | Why it happens | Fix |
|---|---|---|
| Random train/test split | Muscle memory from standard ML workflows | Always use walk-forward or a strict temporal split |
| Using future features | Off-by-one error when building lag features | Explicitly audit every feature's timestamp against the target's timestamp |
| Overfitting to seasonality | Model memorizes the calendar instead of learning general structure | Hold out at least one full seasonal cycle in the test set |
| Ignoring scale changes | Absolute values grow (e.g., revenue doubles) but the underlying pattern shape doesn't | Model percentage/relative change instead of absolute values |
| Too many lag features | Assuming "more history = better" | Use ACF/PACF to justify each lag you include |
| Skipping differencing | Assuming the model will "figure out" the trend on its own | Tree models can handle trend implicitly; linear/AR models generally need a stationary input |

---

## 12. Practical Checklist Before You Model Anything

1. **Plot the raw series first.** Look for trend, seasonality, outliers, and structural breaks. This alone often reveals more than an hour of automated diagnostics.
2. **Difference before lagging, if there's a clear trend.** Never hurts, and is required for linear/AR-style models.
3. **Hold out at least one full seasonal cycle** in your test set, or you can't actually verify the model learned the seasonal pattern.
4. **Monitor in production.** Time series models degrade as the world changes; track rolling prediction error and retrain when it climbs.
5. **Watch for regime changes** (e.g., pre- vs. post-pandemic behavior). Either add explicit regime indicators as features or use a sliding window that forgets stale data.
6. **Log-transform right-skewed series** (revenue, prices, counts) to stabilize variance; forecast in log space, then exponentiate back.

---

## Key Terms

| Term | Casual description | Precise meaning |
|---|---|---|
| Stationarity | "The stats don't change over time" | Mean, variance, and autocorrelation structure are constant over time |
| Differencing | "Subtract consecutive values" | Computing `y[t] - y[t-1]` to remove trend and move toward stationarity |
| ACF | "How a series correlates with itself" | Correlation between a series and a lagged copy of itself, as a function of lag |
| PACF | "Direct correlation only" | Autocorrelation at lag k after removing the effect of all shorter lags |
| Lag features | "Past values as inputs" | Using y[t-1] … y[t-k] as predictors for y[t] |
| Walk-forward validation | "Time-respecting cross-validation" | Evaluation where training data always precedes test data chronologically |
| ARIMA | "The classic time series model" | Combines autoregression (AR), differencing (I), and past-error correction (MA) |
| Seasonality | "Repeating calendar patterns" | Regular, predictable cycles tied to calendar periods |
| Trend | "The long-term direction" | Persistent increase or decrease in the series' level over time |
| Expanding window | "Use all history" | Training set that grows with each walk-forward fold |
| Sliding window | "Fixed-size history" | Training set that stays a fixed length and slides forward each fold |

## Further Reading

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.) — https://otexts.com/fpp3/
- scikit-learn `TimeSeriesSplit` docs — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- statsmodels ARIMA docs — https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
- Makridakis et al., *The M5 Competition* (2022) — https://www.sciencedirect.com/science/article/pii/S0169207021001874
