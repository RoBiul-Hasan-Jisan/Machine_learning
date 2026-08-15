"""

Rebalance a training set when one class vastly outnumbers another. All
four functions here operate on (X, y) arrays and should be applied to
TRAINING data ONLY, after splitting off a test set -- resampling before
the split lets duplicated or synthetic rows leak across the train/test
boundary and produces a dishonest, overly-optimistic evaluation score.
"""

import numpy as np



# Random Oversampling


def random_oversample(X, y, random_state=42):
    """
    Duplicate minority-class rows (with replacement) until every class
    has as many samples as the majority class. Simple and fast, but
    since it's exact duplication, a model can effectively "memorize"
    the repeated minority points rather than learning a general
    decision boundary -- higher overfitting risk than SMOTE.
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_parts, y_parts = [X], [y]
    for cls, count in zip(classes, counts):
        if count == max_count:
            continue
        cls_idx = np.where(y == cls)[0]
        extra_idx = rng.choice(cls_idx, size=max_count - count, replace=True)
        X_parts.append(X[extra_idx])
        y_parts.append(y[extra_idx])

    return np.vstack(X_parts), np.concatenate(y_parts)



# Random Undersampling


def random_undersample(X, y, random_state=42):
    """
    Remove majority-class rows (randomly, without replacement) until
    every class matches the minority class's count. Simple, but throws
    away data -- can hurt if the majority class had useful diversity the
    model needed to see.
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_parts, y_parts = [], []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        keep_idx = rng.choice(cls_idx, size=min_count, replace=False)
        X_parts.append(X[keep_idx])
        y_parts.append(y[keep_idx])

    return np.vstack(X_parts), np.concatenate(y_parts)



# SMOTE


def _k_nearest(point, candidates, k):
    dists = np.sqrt(((candidates - point) ** 2).sum(axis=1))
    return np.argsort(dists)[:k]


def smote(X, y, minority_class=None, k_neighbors=5, random_state=42):
    """
    Synthetic Minority Oversampling Technique. For each minority-class
    point, pick one of its k nearest MINORITY-class neighbors and
    generate a new synthetic point somewhere on the line segment
    between them:

        synthetic = point + random(0,1) * (neighbor - point)

    This avoids exact duplication (unlike random_oversample) by
    inventing plausible new points rather than copying existing ones --
    but if the minority class is sparse or noisy, "plausible" line
    segments between distant points can land in an unrealistic region
    of feature space.

    Args:
        minority_class: which class label to oversample; if None,
            auto-detects the class with the fewest samples.
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)

    if minority_class is None:
        minority_class = classes[np.argmin(counts)]

    majority_count = counts.max()
    minority_mask = y == minority_class
    X_minority = X[minority_mask]
    n_minority = len(X_minority)
    n_to_generate = majority_count - n_minority

    if n_to_generate <= 0:
        return X, y

    k = min(k_neighbors, n_minority - 1)
    synthetic_points = []

    for _ in range(n_to_generate):
        i = rng.randint(0, n_minority)
        point = X_minority[i]
        neighbor_idx = _k_nearest(point, X_minority, k + 1)
        neighbor_idx = neighbor_idx[neighbor_idx != i]  # exclude the point itself
        chosen_neighbor = X_minority[rng.choice(neighbor_idx)]

        gap = rng.uniform(0, 1)
        synthetic_points.append(point + gap * (chosen_neighbor - point))

    X_synthetic = np.array(synthetic_points)
    y_synthetic = np.full(n_to_generate, minority_class)

    return np.vstack([X, X_synthetic]), np.concatenate([y, y_synthetic])



# ADASYN


def adasyn(X, y, minority_class=None, k_neighbors=5, random_state=42):
    """
    Adaptive Synthetic Sampling: like SMOTE, but generates MORE synthetic
    points for minority samples that are surrounded by more majority-class
    neighbors (i.e. points near the decision boundary, where the classes
    are hardest to tell apart) and fewer synthetic points for minority
    samples already deep in a minority-dense region.

    Mechanism:
      1. For each minority point, look at its k nearest neighbors
         (across BOTH classes) and compute r_i = (# majority neighbors) / k.
         A higher r_i means this point is in a more contested, harder
         region of feature space.
      2. Normalize r_i across all minority points so they sum to 1 --
         this becomes each point's SHARE of the total synthetic budget.
      3. Generate synthetic points the same way SMOTE does (interpolate
         toward a same-class neighbor), but allocate more of the budget
         to high-r_i points.

    This focuses modeling effort where the boundary is genuinely
    ambiguous -- but for the same reason, it can also amplify noise if a
    minority point's "hard" status is due to a labeling error rather
    than genuine class overlap.
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)

    if minority_class is None:
        minority_class = classes[np.argmin(counts)]

    majority_count = counts.max()
    minority_mask = y == minority_class
    X_minority = X[minority_mask]
    n_minority = len(X_minority)
    n_to_generate = majority_count - n_minority

    if n_to_generate <= 0:
        return X, y

    k = min(k_neighbors, len(X) - 1)

    # step 1: r_i = fraction of each minority point's k-NN (over ALL classes) that are majority
    r = np.zeros(n_minority)
    for i, point in enumerate(X_minority):
        neighbor_idx = _k_nearest(point, X, k + 1)
        # the point itself is in X too; drop the zero-distance self match if present
        self_matches = np.where(np.all(X[neighbor_idx] == point, axis=1))[0]
        if len(self_matches) > 0:
            neighbor_idx = np.delete(neighbor_idx, self_matches[0])
        neighbor_idx = neighbor_idx[:k]
        r[i] = np.mean(y[neighbor_idx] != minority_class)

    # step 2: normalize into a synthetic-sample allocation
    if r.sum() == 0:
        r_norm = np.full(n_minority, 1.0 / n_minority)  # fallback: uniform, like SMOTE
    else:
        r_norm = r / r.sum()
    n_per_point = np.round(r_norm * n_to_generate).astype(int)

    # step 3: generate, same interpolation mechanism as SMOTE, weighted by n_per_point
    k_within = min(k_neighbors, n_minority - 1)
    synthetic_points = []
    for i, n_gen in enumerate(n_per_point):
        if n_gen == 0:
            continue
        point = X_minority[i]
        neighbor_idx = _k_nearest(point, X_minority, k_within + 1)
        neighbor_idx = neighbor_idx[neighbor_idx != i]
        for _ in range(n_gen):
            chosen_neighbor = X_minority[rng.choice(neighbor_idx)]
            gap = rng.uniform(0, 1)
            synthetic_points.append(point + gap * (chosen_neighbor - point))

    if not synthetic_points:
        return X, y

    X_synthetic = np.array(synthetic_points)
    y_synthetic = np.full(len(X_synthetic), minority_class)

    return np.vstack([X, X_synthetic]), np.concatenate([y, y_synthetic])





def _demo():
    
    print("  IMBALANCED DATA DEMO")
   

    rng = np.random.RandomState(0)
    n_majority, n_minority = 100, 10

    X_majority = rng.normal(0, 1, (n_majority, 2))
    X_minority = rng.normal(2, 0.7, (n_minority, 2))  # somewhat overlapping with majority
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * n_majority + [1] * n_minority)

    print(f"\nOriginal class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_over, y_over = random_oversample(X, y, random_state=0)
    print(f"After random oversampling:   {dict(zip(*np.unique(y_over, return_counts=True)))}")

    X_under, y_under = random_undersample(X, y, random_state=0)
    print(f"After random undersampling:  {dict(zip(*np.unique(y_under, return_counts=True)))}")

    X_smote, y_smote = smote(X, y, k_neighbors=5, random_state=0)
    print(f"After SMOTE:                  {dict(zip(*np.unique(y_smote, return_counts=True)))}")
    n_synth = len(X_smote) - len(X)
    print(f"  ({n_synth} synthetic points generated, interpolated between real minority points)")

    X_adasyn, y_adasyn = adasyn(X, y, k_neighbors=5, random_state=0)
    print(f"After ADASYN:                 {dict(zip(*np.unique(y_adasyn, return_counts=True)))}")

    print("\nNote: SMOTE spreads new points roughly evenly across the minority")
    print("class; ADASYN concentrates more new points near minority samples")
    print("whose neighbors are mostly majority-class (i.e. near the boundary).")
    print("\nReminder: fit resamplers on the TRAINING split only, after your")
    print("train/test split -- never on the full dataset before splitting.")


if __name__ == "__main__":
    _demo()
