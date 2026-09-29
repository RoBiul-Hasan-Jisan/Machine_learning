""" TARNet (shared-body two-head net) vs a fully separate T-learner (both NumPy MLPs)."""
import numpy as np

rng = np.random.default_rng(0)

def simulate(n):
    X = rng.normal(0, 1, (n, 5))
    e = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    T = rng.binomial(1, e)
    mu0 = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2]
    tau = 1.0 + 1.5 * X[:, 0]
    Y = mu0 + T * tau + rng.normal(0, 0.5, n)
    return X, T, Y, tau

class MLP:
    """One-hidden-layer MLP trained by full-batch gradient descent."""
    def __init__(self, d_in, d_hidden, seed):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.5, (d_in, d_hidden)); self.b1 = np.zeros(d_hidden)
        self.W2 = r.normal(0, 0.5, d_hidden); self.b2 = 0.0
    def forward(self, X):
        self.H = np.maximum(0, X @ self.W1 + self.b1)
        return self.H @ self.W2 + self.b2
    def step(self, X, y, lr):
        pred = self.forward(X); err = pred - y
        gW2 = self.H.T @ err / len(y); gb2 = err.mean()
        gH = np.outer(err, self.W2) * (self.H > 0) / len(y)
        gW1 = X.T @ gH; gb1 = gH.sum(0)
        self.W2 -= lr * gW2; self.b2 -= lr * gb2; self.W1 -= lr * gW1; self.b1 -= lr * gb1
        return np.mean(err ** 2)

def train_shared_body(X, T, Y, hidden=16, epochs=400, lr=0.05):
    """TARNet: shared hidden layer, two linear output heads."""
    d = X.shape[1]
    r = np.random.default_rng(0)
    W1 = r.normal(0, 0.5, (d, hidden)); b1 = np.zeros(hidden)
    W2 = {0: r.normal(0, 0.5, hidden), 1: r.normal(0, 0.5, hidden)}
    b2 = {0: 0.0, 1: 0.0}
    for _ in range(epochs):
        H = np.maximum(0, X @ W1 + b1)
        for t in (0, 1):
            m = T == t
            pred = H[m] @ W2[t] + b2[t]
            err = pred - Y[m]
            gW2 = H[m].T @ err / m.sum(); gb2 = err.mean()
            gH = np.outer(err, W2[t]) * (H[m] > 0) / m.sum()
            W2[t] -= lr * gW2; b2[t] -= lr * gb2
            W1 -= lr * (X[m].T @ gH); b1 -= lr * gH.sum(0)
    def predict(Xnew):
        Hn = np.maximum(0, Xnew @ W1 + b1)
        return Hn @ W2[1] + b2[1], Hn @ W2[0] + b2[0]
    return predict

X, T, Y, tau = simulate(4000)
Xte, Tte, Yte, tau_te = simulate(2000)

# Fully separate T-learner: independent MLPs, no shared body
m1, m0 = MLP(5, 16, 1), MLP(5, 16, 2)
for _ in range(400):
    m1.step(X[T == 1], Y[T == 1], 0.05); m0.step(X[T == 0], Y[T == 0], 0.05)
tau_sep = m1.forward(Xte) - m0.forward(Xte)

tarnet_predict = train_shared_body(X, T, Y)
pred1, pred0 = tarnet_predict(Xte)
tau_tarnet = pred1 - pred0

print(f"True ATE (test) = {tau_te.mean():.3f}\n")
print(f"{'model':<22}{'ATE est':>9}{'RMSE':>8}{'corr w/ true tau':>18}")
for name, th in (("Separate T-learner", tau_sep), ("TARNet (shared body)", tau_tarnet)):
    print(f"{name:<22}{th.mean():>9.3f}{np.sqrt(np.mean((th - tau_te) ** 2)):>8.3f}{np.corrcoef(th, tau_te)[0, 1]:>18.3f}")
