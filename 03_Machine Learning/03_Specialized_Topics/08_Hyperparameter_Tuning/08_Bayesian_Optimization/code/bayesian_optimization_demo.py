

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(max_depth_float):
    """The 'black box' function we're optimizing: CV ROC-AUC as a function of max_depth."""
    max_depth = int(round(max_depth_float))
    model = RandomForestClassifier(n_estimators=150, max_depth=max_depth, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()


def expected_improvement(X_candidates, gp_model, best_so_far, xi=0.01):
    """Acquisition function: how promising is each candidate point?"""
    mu, sigma = gp_model.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_so_far - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return ei


# Search space: max_depth from 2 to 30 (treated as continuous for the GP, rounded when evaluated)
search_space = np.linspace(2, 30, 300).reshape(-1, 1)

# --- Step 1: a few random initial trials to seed the surrogate model ---
np.random.seed(42)
X_observed = np.random.uniform(2, 30, size=4).reshape(-1, 1)
y_observed = np.array([evaluate(x[0]) for x in X_observed])

print("Initial random trials:")
for x, score in zip(X_observed, y_observed):
    print(f"  max_depth={x[0]:.1f} -> CV ROC-AUC={score:.4f}")

n_bayesian_trials = 8
for trial in range(n_bayesian_trials):
    # --- Step 2: fit the surrogate model on all trials so far (normalize inputs for GP stability) ---
    X_norm = (X_observed - 2) / (30 - 2)
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5, length_scale=0.2, length_scale_bounds=(0.01, 1.0)),
                                   normalize_y=True, random_state=42, n_restarts_optimizer=3)
    gp.fit(X_norm, y_observed)

    # --- Step 3: use the acquisition function to pick the next candidate ---
    search_space_norm = (search_space - 2) / (30 - 2)
    ei_values = expected_improvement(search_space_norm, gp, y_observed.max(), xi=0.005)

    # Avoid re-picking a point too close to one already tried
    for tried_x in X_observed.ravel():
        too_close = np.abs(search_space.ravel() - tried_x) < 0.5
        ei_values[too_close] = -np.inf

    next_x = search_space[np.argmax(ei_values)]

    # --- Step 4: actually evaluate that candidate ---
    next_y = evaluate(next_x[0])

    X_observed = np.vstack([X_observed, next_x])
    y_observed = np.append(y_observed, next_y)

    print(f"Trial {trial+1}: chose max_depth={next_x[0]:.1f} -> CV ROC-AUC={next_y:.4f} "
          f"(best so far: {y_observed.max():.4f})")

best_idx = np.argmax(y_observed)
best_depth = int(round(X_observed[best_idx][0]))
print(f"\nBest max_depth found: {best_depth} (CV ROC-AUC = {y_observed[best_idx]:.4f})")

final_model = RandomForestClassifier(n_estimators=150, max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)
print(f"Test accuracy with tuned max_depth: {final_model.score(X_test, y_test):.4f}")

print("\nNote how few total trials (4 random + 8 Bayesian = 12) were needed to converge")
print("on a strong region -- a comparable Grid Search over the same range with a fine")
print("enough step size would need many more evaluations to be this confident.")
