# Ridge, Lasso, and ElasticNet Regression

## Learning Objectives

- Understand why unregularized linear regression overfits and how L1/L2 penalties fix it
- Implement Ridge (L2), Lasso (L1), and ElasticNet (L1 + L2) regression from scratch using gradient descent
- Explain why Lasso produces sparse (zeroed-out) coefficients and Ridge does not
- Choose between Ridge, Lasso, and ElasticNet based on the data and the goal (prediction vs feature selection)

## The Problem

Plain linear regression minimizes squared error:

```
Loss(w) = sum((y_i - X_i . w)^2)
```

With enough features, or with correlated features, this overfits. The model can chase noise by assigning huge weights to features that happen to correlate with the target in the training set. Small changes in the training data then produce wildly different coefficients, and test performance suffers.

Regularization fixes this by penalizing large coefficients. You trade a little bias for a lot less variance. Ridge, Lasso, and ElasticNet are the three standard ways to do this, and they differ only in which penalty they add to the loss.

## The Concept

### Ridge regression (L2 penalty)

Ridge adds the sum of squared coefficients to the loss:

```
Loss(w) = sum((y_i - X_i . w)^2) + alpha * sum(w_j^2)
```

`alpha` controls the strength of the penalty. `alpha = 0` recovers ordinary least squares. As `alpha` grows, coefficients shrink toward zero, but rarely hit exactly zero.

Ridge is useful when you believe most features are at least a little relevant, and you mainly want to control variance and handle multicollinearity (correlated features). It shrinks correlated features together rather than picking one over another.

### Lasso regression (L1 penalty)

Lasso adds the sum of absolute coefficients:

```
Loss(w) = sum((y_i - X_i . w)^2) + alpha * sum(|w_j|)
```

The key difference from Ridge: the L1 penalty has a corner at zero. Geometrically, this means the optimal solution often lands exactly on a coefficient of zero, not just close to it. Lasso performs automatic feature selection: irrelevant features get a coefficient of exactly 0, effectively removing them from the model.

```
Why L1 zeroes coefficients and L2 does not (intuition):

L2 penalty gradient near w=0: proportional to w itself -> shrinks smoothly, slows down near 0
L1 penalty gradient near w=0: constant magnitude (+alpha or -alpha) -> keeps pushing toward
exactly 0 until it gets there, then stays there (subgradient allows w=0 to be optimal)
```

Lasso is useful when you suspect only a subset of features actually matter, or when you want a sparse, interpretable model.

### ElasticNet (L1 + L2 combined)

ElasticNet combines both penalties:

```
Loss(w) = sum((y_i - X_i . w)^2) + alpha * (l1_ratio * sum(|w_j|) + (1 - l1_ratio) * sum(w_j^2))
```

`l1_ratio` (between 0 and 1) controls the mix. `l1_ratio = 1` is pure Lasso. `l1_ratio = 0` is pure Ridge.

ElasticNet exists because Lasso has a weakness: when features are highly correlated, Lasso tends to arbitrarily pick one and zero out the rest, which can be unstable. ElasticNet's Ridge component encourages correlated features to be selected (or shrunk) together, while the Lasso component still produces sparsity.

| Method | Penalty | Coefficients | Best for |
|--------|---------|--------------|----------|
| Ridge | L2: sum(w_j^2) | Shrunk, rarely exactly 0 | Multicollinearity, many mildly-relevant features |
| Lasso | L1: sum\|w_j\| | Sparse, many exactly 0 | Feature selection, few truly relevant features |
| ElasticNet | Mix of L1 and L2 | Sparse, correlated features grouped | Correlated features + need for sparsity |

### Choosing alpha (and l1_ratio)

`alpha` is a hyperparameter, not learned from the loss. It is chosen by cross-validation: try a range of values (often on a log scale, e.g. 0.001, 0.01, 0.1, 1, 10, 100), measure validation error for each, and pick the one that generalizes best.

```
alpha too small  -> behaves like plain linear regression -> overfits
alpha too large  -> all coefficients shrink toward 0 -> underfits
```

For ElasticNet, you cross-validate over a grid of both `alpha` and `l1_ratio`.

### Why gradient descent for Lasso

Ridge has a closed-form solution (it's just a modified normal equation), but Lasso's L1 penalty is not differentiable at `w = 0`, so there's no clean closed form. In practice, Lasso is solved with coordinate descent or subgradient methods. This lesson uses (sub)gradient descent for all three so the relationship between them is easy to see in code.

Feature scaling matters here: since the penalty is applied uniformly to all coefficients, features on larger scales will be penalized unfairly relative to features on smaller scales. Always standardize features (zero mean, unit variance) before fitting.

## Build It

### Step 1: Ridge regression from scratch

```python
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iters=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error) + 2 * self.alpha * self.weights
            grad_b = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias
```

Note the gradient of the penalty term `alpha * sum(w_j^2)` is `2 * alpha * w_j`, which is why it shows up as `2 * self.alpha * self.weights` and shrinks weights proportionally to their current size.

### Step 2: Lasso regression from scratch

The L1 penalty is not differentiable at 0, so we use its subgradient: `sign(w_j)` (which is -1, 0, or 1).

```python
import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iters=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error) + self.alpha * np.sign(self.weights)
            grad_b = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias
```

Because `sign(w_j)` is constant regardless of how small `w_j` is (as long as it's nonzero), gradient descent keeps pushing small weights all the way to zero rather than slowing down near it, which is what produces sparsity.

### Step 3: ElasticNet from scratch

Combine both gradients, weighted by `l1_ratio`.

```python
import numpy as np

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, n_iters=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            l1_grad = self.l1_ratio * np.sign(self.weights)
            l2_grad = (1 - self.l1_ratio) * 2 * self.weights
            grad_w = (2 / n_samples) * (X.T @ error) + self.alpha * (l1_grad + l2_grad)
            grad_b = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias
```

Setting `l1_ratio=1.0` reduces this exactly to Lasso; `l1_ratio=0.0` reduces it exactly to Ridge.

### Step 4: Standardize features first

All three penalties assume features are on comparable scales.

```python
import numpy as np

def standardize(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant columns
    return (X - mean) / std, mean, std
```

See `code/regularized_regression.py` for the complete implementation with all three classes plus the standardization helper.

## Use It

With scikit-learn, all three are one-liners:

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
lasso = Lasso(alpha=0.1).fit(X_train_s, y_train)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train_s, y_train)

print("Ridge R^2:", ridge.score(X_test_s, y_test))
print("Lasso R^2:", lasso.score(X_test_s, y_test))
print("Lasso nonzero coefs:", (lasso.coef_ != 0).sum(), "/", len(lasso.coef_))
print("ElasticNet R^2:", elastic.score(X_test_s, y_test))
```

For choosing `alpha` automatically, scikit-learn provides `RidgeCV`, `LassoCV`, and `ElasticNetCV`, which run cross-validation internally over a grid of alpha values.

## Exercises

1. Generate a dataset with 20 features where only 5 are actually informative (the rest are noise). Fit Lasso with increasing `alpha` and plot how many coefficients become exactly zero. Verify Lasso recovers close to the 5 informative features.

2. Create two highly correlated features (e.g. `x2 = x1 + small_noise`) and fit Lasso vs Ridge vs ElasticNet. Observe that Lasso arbitrarily zeroes one of the pair while Ridge and ElasticNet shrink both together.

3. Implement a manual grid search over `alpha` (e.g. 0.001 to 100 on a log scale) using a validation split, and plot validation error vs alpha for Ridge. Identify the U-shaped curve: underfitting on one end, overfitting on the other.

4. Compare Ridge, Lasso, and unregularized linear regression on a dataset with more features than samples (n_features > n_samples). Explain why unregularized regression fails here and regularization does not.

5. Implement coordinate descent for Lasso (updating one coefficient at a time using the soft-thresholding operator) and compare convergence speed against the subgradient descent version in this lesson.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Regularization | "Penalizing complexity" | Adding a penalty term to the loss function that discourages large coefficients, trading bias for reduced variance |
| Ridge regression | "L2 regularization" | Adds sum of squared coefficients to the loss. Shrinks coefficients smoothly, rarely to exactly zero |
| Lasso regression | "L1 regularization" | Adds sum of absolute coefficients to the loss. Produces sparse solutions with many coefficients exactly zero |
| ElasticNet | "Best of both" | Combines L1 and L2 penalties, controlled by `l1_ratio`. Sparse like Lasso, but groups correlated features like Ridge |
| alpha (lambda) | "Regularization strength" | The hyperparameter controlling how much the penalty term matters relative to the fit. Chosen via cross-validation |
| l1_ratio | "The L1/L2 mix" | ElasticNet's mixing parameter between 0 (pure Ridge) and 1 (pure Lasso) |
| Sparsity | "Most coefficients are zero" | A model where many coefficients are exactly zero, effectively performing feature selection |
| Multicollinearity | "Features that overlap" | When two or more features are highly correlated with each other, making individual coefficient estimates unstable |
| Coordinate descent | "Update one at a time" | An optimization method that updates one coefficient at a time while holding others fixed; the standard solver for Lasso |
| Feature standardization | "Put features on the same scale" | Rescaling features to zero mean and unit variance, required before regularization so penalties are applied fairly |

