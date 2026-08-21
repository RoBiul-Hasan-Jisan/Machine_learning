"""
Single-step and multi-step (direct and recursive) time series
forecasting with an LSTM, correct chronological splitting,
training-set-only normalization, and comparison against a naive baseline.
"""

import numpy as np
import torch
import torch.nn as nn


def make_synthetic_series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 24)      # daily-ish cycle
    trend = 0.02 * t
    noise = rng.normal(0, 1.0, size=n)
    return seasonal + trend + noise + 50.0


def make_sliding_windows(series, window_size, horizon=1):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size:i + window_size + horizon])
    return np.array(X), np.array(y)


class ForecastLSTM(nn.Module):
    def __init__(self, hidden_size=32, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, window_size) -> (batch, window_size, 1)
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def train_model(model, X_train, y_train, n_epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}: loss = {loss.item():.4f}")


def demo_single_step_forecast():
    print("=== Single-step forecasting ===")
    series = make_synthetic_series()
    cutoff = int(len(series) * 0.8)
    train_series, test_series = series[:cutoff], series[cutoff:]

    train_mean, train_std = train_series.mean(), train_series.std()
    train_norm = (train_series - train_mean) / train_std
    test_norm = (test_series - train_mean) / train_std  # training stats only

    window_size = 12
    X_train, y_train = make_sliding_windows(train_norm, window_size, horizon=1)
    X_test, y_test = make_sliding_windows(test_norm, window_size, horizon=1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = ForecastLSTM(horizon=1)
    train_model(model, X_train_t, y_train_t)

    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_t).numpy()

    preds = preds_norm * train_std + train_mean  # un-normalize
    y_test_actual = y_test * train_std + train_mean

    model_mae = mae(y_test_actual, preds)

    # Naive baseline: predict "same as the last value in the window"
    naive_preds = X_test[:, -1:] * train_std + train_mean
    naive_mae = mae(y_test_actual, naive_preds)

    print(f"\nModel MAE:  {model_mae:.4f}")
    print(f"Naive baseline MAE (predict last value): {naive_mae:.4f}")
    if model_mae < naive_mae:
        print("Model beats the naive baseline.\n")
    else:
        print("Model does NOT beat the naive baseline -- would need more training/tuning.\n")


def demo_multistep_forecast():
    print("=== Multi-step forecasting: direct vs recursive ===")
    series = make_synthetic_series(seed=1)
    cutoff = int(len(series) * 0.8)
    train_series, test_series = series[:cutoff], series[cutoff:]

    train_mean, train_std = train_series.mean(), train_series.std()
    train_norm = (train_series - train_mean) / train_std
    test_norm = (test_series - train_mean) / train_std

    window_size = 12
    horizon = 5

    # --- Direct multi-step ---
    X_train, y_train = make_sliding_windows(train_norm, window_size, horizon=horizon)
    X_test, y_test = make_sliding_windows(test_norm, window_size, horizon=horizon)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    direct_model = ForecastLSTM(horizon=horizon)
    print("Training DIRECT multi-step model...")
    train_model(direct_model, X_train_t, y_train_t)

    direct_model.eval()
    with torch.no_grad():
        direct_preds_norm = direct_model(X_test_t).numpy()
    direct_preds = direct_preds_norm * train_std + train_mean
    y_test_actual = y_test * train_std + train_mean

    # --- Recursive multi-step, using the SINGLE-STEP model trained above's architecture ---
    single_step_model = ForecastLSTM(horizon=1)
    X_train_1, y_train_1 = make_sliding_windows(train_norm, window_size, horizon=1)
    print("\nTraining single-step model for RECURSIVE multi-step forecasting...")
    train_model(single_step_model, torch.tensor(X_train_1, dtype=torch.float32),
                torch.tensor(y_train_1, dtype=torch.float32))

    single_step_model.eval()
    recursive_preds = []
    with torch.no_grad():
        for window in X_test:
            current_window = list(window)
            step_preds = []
            for _ in range(horizon):
                inp = torch.tensor([current_window[-window_size:]], dtype=torch.float32)
                next_val = single_step_model(inp).item()
                step_preds.append(next_val)
                current_window.append(next_val)  # feed prediction back in as if real
            recursive_preds.append(step_preds)
    recursive_preds = np.array(recursive_preds) * train_std + train_mean

    print(f"\n{'Step':>6} | {'Direct MAE':>12} | {'Recursive MAE':>14}")
    for step in range(horizon):
        direct_step_mae = mae(y_test_actual[:, step], direct_preds[:, step])
        recursive_step_mae = mae(y_test_actual[:, step], recursive_preds[:, step])
        print(f"{step + 1:6d} | {direct_step_mae:12.4f} | {recursive_step_mae:14.4f}")

    print("\n(Recursive forecasting's error typically grows faster across steps,")
    print("since later predictions are built on top of earlier ones that may")
    print("already be wrong -- this is error accumulation in action.)")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_single_step_forecast()

    torch.manual_seed(0)
    demo_multistep_forecast()
