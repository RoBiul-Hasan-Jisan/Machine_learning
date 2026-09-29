""" CATE: S-learner vs T-learner (randomized data, known true effects)."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
n = 8_000
X = rng.normal(0, 1, (n, 3))
T = rng.binomial(1, 0.5, n)                       # randomized
tau = 1 + 2 * X[:, 0]
Y0 = X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 1, n)
Y = np.where(T == 1, Y0 + tau, Y0)

train = np.arange(n) < 6_000
test = ~train
gbr = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)

# S-learner
Fs = np.column_stack([X, T])
s = gbr().fit(Fs[train], Y[train])
tau_s = s.predict(np.column_stack([X[test], np.ones(test.sum())])) - s.predict(np.column_stack([X[test], np.zeros(test.sum())]))

# T-learner
m1 = gbr().fit(X[train & (T == 1)], Y[train & (T == 1)])
m0 = gbr().fit(X[train & (T == 0)], Y[train & (T == 0)])
tau_t = m1.predict(X[test]) - m0.predict(X[test])

print(f"True ATE (test) = {tau[test].mean():.3f}\n")
print(f"{'learner':<12}{'ATE est':>9}{'RMSE':>8}{'corr(tau_hat, tau)':>22}")
for name, th in (("S-learner", tau_s), ("T-learner", tau_t)):
    print(f"{name:<12}{th.mean():>9.3f}{np.sqrt(np.mean((th - tau[test]) ** 2)):>8.3f}{np.corrcoef(th, tau[test])[0, 1]:>22.3f}")

print("\nAverage effect by true-tau quartile (true vs T-learner estimate):")
q = np.quantile(tau[test], [0.25, 0.5, 0.75]); g = np.digitize(tau[test], q)
for k in range(4):
    print(f"  quartile {k+1}: true {tau[test][g == k].mean():6.2f}   estimated {tau_t[g == k].mean():6.2f}")
