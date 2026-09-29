""" Double/debiased ML vs naive approaches on a nonlinear-confounded partially linear model."""
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

THETA = 1.0
rf = lambda: RandomForestRegressor(n_estimators=60, min_samples_leaf=5, n_jobs=-1, random_state=0)

def simulate(n, rng):
    X = rng.normal(0, 1, (n, 10))
    m = np.cos(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 2]
    g = 2 * np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 3] * X[:, 4]
    T = m + rng.normal(0, 1, n)
    Y = THETA * T + g + rng.normal(0, 1, n)
    return X, T, Y

def dml(X, T, Y, cross_fit):
    n = len(Y)
    if cross_fit:
        l_hat, m_hat = np.zeros(n), np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=1).split(X):
            l_hat[te] = rf().fit(X[tr], Y[tr]).predict(X[te])
            m_hat[te] = rf().fit(X[tr], T[tr]).predict(X[te])
    else:
        l_hat = rf().fit(X, Y).predict(X); m_hat = rf().fit(X, T).predict(X)
    yr, vr = Y - l_hat, T - m_hat
    return np.sum(vr * yr) / np.sum(vr ** 2)

def estimates(X, T, Y):
    naive = np.cov(T, Y)[0, 1] / np.var(T, ddof=1)
    D = np.column_stack([np.ones(len(Y)), T, X]); linear = np.linalg.lstsq(D, Y, rcond=None)[0][1]
    plug = rf().fit(np.column_stack([X, T]), Y)                      # RF plug-in with T as a feature
    t1, t0 = np.column_stack([X, np.ones(len(Y))]), np.column_stack([X, np.zeros(len(Y))])
    plugin = np.mean(plug.predict(t1) - plug.predict(t0))
    return [naive, linear, plugin, dml(X, T, Y, False), dml(X, T, Y, True)]

rng = np.random.default_rng(0)
start = time.time()
res = np.array([estimates(*simulate(2000, rng)) for _ in range(10)])
names = ["naive Y~T", "linear adjustment", "RF plug-in", "DML no cross-fit", "DML cross-fit"]
print(f"True theta = {THETA}   (10 repeats, n=2000; {time.time() - start:.0f}s)\n")
print(f"{'estimator':<20}{'mean':>8}{'bias':>8}{'std':>8}")
for name, col in zip(names, res.T):
    print(f"{name:<20}{col.mean():>8.3f}{col.mean() - THETA:>+8.3f}{col.std():>8.3f}")
