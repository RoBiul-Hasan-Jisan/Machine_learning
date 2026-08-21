# 15. Time Series Forecasting

## Learning Objectives

- Frame a time series forecasting problem as a supervised learning task an RNN can train on
- Implement a sliding-window training setup and both single-step and multi-step forecasting
- Apply correct time-series-specific evaluation (no random shuffling) and interpret common forecasting metrics

## The Problem

Every task so far in this module has been about language — text classification, translation, generation. RNNs are just as naturally suited to a very different kind of sequential data: numeric time series (stock prices, sensor readings, energy demand, weather). The core mechanism (Lessons 03-08) transfers directly, but framing the *problem* correctly — how training examples are constructed, how forecasting multiple steps ahead works, and how to evaluate honestly — has task-specific considerations this lesson covers.

## The Concept

### Framing forecasting as supervised learning: the sliding window

A raw time series is just one long sequence of numbers. To train a model, it needs to be converted into (input, target) pairs — a **sliding window** approach uses a fixed-length window of past values to predict the next value(s), stepping the window forward by one position to generate each training example:

```
Raw series:  [10, 12, 11, 15, 14, 18, 17, 20, ...]

Window size 3, predicting 1 step ahead:

Example 1:  input = [10, 12, 11]  ->  target = 15
Example 2:  input = [12, 11, 15]  ->  target = 14
Example 3:  input = [11, 15, 14]  ->  target = 18
Example 4:  input = [15, 14, 18]  ->  target = 17
...
```

```python
def make_sliding_windows(series, window_size, horizon=1):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + horizon])
    return np.array(X), np.array(y)
```

Each window becomes one training sequence, fed through the RNN exactly as in Lesson 04's many-to-one pattern: the RNN reads the window, and the final hidden state is used to predict the target value(s).

### Single-step vs multi-step forecasting

**Single-step**: predict only the very next value (`horizon=1` above). Straightforward, and the RNN's many-to-one pattern applies directly.

**Multi-step**: predict several future values at once (`horizon=5`, say). Two common strategies:

```
Direct multi-step:      the RNN's final hidden state feeds a FC layer with `horizon` outputs,
                         predicting all future steps in ONE shot from the same hidden state

Recursive multi-step:   predict one step ahead, then feed that PREDICTION back in as if it were
                         real data, predict the next step, and repeat -- exactly Lesson 11's
                         decoder pattern, applied to numbers instead of tokens
```

Recursive multi-step is more flexible (works for any horizon without retraining) but suffers from **error accumulation**: an error in an early predicted step becomes part of the input for predicting the next step, and errors can compound the further out the forecast goes — the numeric analogue of Lesson 11's exposure bias. Direct multi-step avoids this compounding (every step is predicted from real, not predicted, history) but needs a fixed horizon decided in advance and doesn't share information as naturally between the steps it predicts.

### Normalization matters, and must be fit on training data only

Time series often have very different scales (temperature in the tens, stock prices in the hundreds, sensor voltages near zero) — feeding raw values directly can make training unstable, similar to why the CNN module's transfer learning lesson insisted on specific normalization. Normalize (e.g. z-score using the *training* portion's mean and standard deviation) before creating sliding windows, and remember to un-normalize predictions before interpreting or reporting them:

```python
train_mean, train_std = train_series.mean(), train_series.std()
train_normalized = (train_series - train_mean) / train_std
test_normalized = (test_series - train_mean) / train_std   # use TRAINING stats, not test stats
```

Using the test set's own statistics to normalize the test set is a direct instance of the leakage pattern general ML practice warns against: it lets information about the test distribution influence preprocessing before evaluation.

### Evaluating time series models: never shuffle

This is the single most important, easy-to-get-wrong detail specific to time series: **never randomly shuffle time series data into train/test splits.** A time series has a genuine temporal order, and shuffling would let the model train on data from *after* the test period, which is a direct form of the time leakage the classical ML curriculum's train/validation/test lesson warned about generally. Instead, split chronologically:

```
WRONG:  random_train_test_split(all_windows)      -- can put "future" data in training

RIGHT:  train = series[:cutoff_date]
        test  = series[cutoff_date:]               -- test is strictly LATER than all training data
```

### Common forecasting metrics

```
MAE  (Mean Absolute Error):        mean(|y_true - y_pred|)              -- same units as the data, easy to interpret
RMSE (Root Mean Squared Error):    sqrt(mean((y_true - y_pred)^2))       -- penalizes large errors more heavily than MAE
MAPE (Mean Absolute Percentage Error): mean(|y_true - y_pred| / |y_true|) * 100   -- scale-independent, but undefined/unstable near y_true=0
```

A crucial sanity check often skipped: compare the model's error against a **naive baseline** — for many time series, simply predicting "tomorrow will be the same as today" (`y_pred = y_true[t-1]`) is a surprisingly strong baseline, and a trained RNN that doesn't clearly beat it is not actually adding value, however good its raw error numbers look in isolation.

See `code/time_series_demo.py` for a complete single-step and multi-step (both direct and recursive) forecasting pipeline on a synthetic seasonal time series, with correct chronological train/test splitting, training-set-only normalization, and a comparison against the naive baseline.

## Exercises

1. Implement `make_sliding_windows` and confirm it produces the correct number of windows for a series of length 100, window size 10, horizon 1.
2. Generate a synthetic time series with a clear repeating seasonal pattern plus noise, split it chronologically, and train an LSTM to forecast one step ahead. Compare its MAE against the naive "same as yesterday" baseline.
3. Implement both direct and recursive multi-step forecasting (horizon=5) on the same synthetic series, and compare their error at each of the 5 forecasted steps — confirm recursive forecasting's error tends to grow faster with the step number, illustrating error accumulation.
4. Deliberately normalize using the *test* set's mean/std instead of the training set's, and compare resulting test MAE against the correct (training-stats-only) version, to see the leakage effect concretely.

## Key Terms

| Term | What it actually means |
|---|---|
| Sliding window | A technique for converting a raw time series into (input, target) training pairs, using a fixed-length window of past values stepped forward one position at a time |
| Horizon | The number of future time steps a forecasting model predicts |
| Direct multi-step forecasting | Predicting all future steps in one shot from the same input window, avoiding error accumulation |
| Recursive multi-step forecasting | Predicting one step, feeding that prediction back in as input, and repeating to reach a longer horizon, risking compounding errors |
| Error accumulation | The tendency of recursive multi-step forecasting's errors to compound, since later predictions depend on earlier (possibly incorrect) predictions |
| Naive baseline | A simple forecasting rule (e.g. predicting the previous value) used as a minimum bar a trained model should clearly outperform |
