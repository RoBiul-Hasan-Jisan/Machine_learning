# Hyperparameter Tuning 

A practical checklist tying together everything from folders 01–12.

## Start simple

Before tuning anything, get a baseline working with sensible defaults. Most libraries' defaults are reasonable starting points — tuning should *improve* on a working baseline, not be the first thing you reach for.

```python
# Step 1: baseline with defaults, just to confirm the pipeline works end-to-end
baseline = RandomForestClassifier(random_state=42)
baseline_score = cross_val_score(baseline, X_train, y_train, cv=5, scoring="roc_auc").mean()
print(f"Baseline CV score: {baseline_score:.4f}")
```

Also start simple in *model choice* — a well-tuned simple model (logistic regression, shallow trees) often beats a poorly-tuned complex one, and gives you a fast, interpretable reference point.

## Use Random Search first

Don't reach for Grid Search or Optuna immediately on a search space you don't understand yet. Random Search (folder 07) with a generous, wide range per hyperparameter is a fast, cheap way to discover **which regions of the space are even promising**, before investing in more expensive, precise methods.

```python
# Step 2: broad Random Search to find promising regions
random_search = RandomizedSearchCV(model, param_distributions=wide_ranges, n_iter=30, cv=5)
random_search.fit(X_train, y_train)
print(random_search.best_params_)  # use this to inform your NEXT search's ranges
```

## Then Optuna

Once Random Search has narrowed down roughly where the good region is, switch to Optuna (folder 09) with **tighter ranges centered on what Random Search found**, and a larger trial budget. Optuna's TPE sampler will refine within that space far more efficiently than continuing to sample randomly or exhaustively gridding it.

```python
# Step 3: Optuna, with narrowed ranges informed by Random Search's best_params_
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),   # narrowed from a wider initial range
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        ...
    }
    ...

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  # can afford more trials now that the space is narrower
```

This progression — **defaults → broad Random Search → narrowed Optuna** — is far more compute-efficient than jumping straight to an expensive exhaustive or Bayesian search over a huge, poorly-understood space.

## Avoid tuning on test data

The single most important rule in this entire chapter, repeated because it's so easy to violate accidentally:

- **Every hyperparameter comparison must happen via cross-validation on training data**, never by checking test set performance and adjusting.
- The test set is touched **exactly once** — after all tuning is finished, to report a final, honest number.
- If you find yourself running the test set more than once during a project, something has gone wrong — treat that as a signal to add a proper validation set or expand your CV strategy instead.

```python
# WRONG -- silently turns the test set into a second validation set
for params in candidate_params:
    model = build_model(**params).fit(X_train, y_train)
    print(model.score(X_test, y_test))  # <- checking test score repeatedly to pick params

# RIGHT -- all comparison happens on CV, test touched once at the very end
best_params = tune_with_cv(X_train, y_train)  # e.g., via RandomizedSearchCV/Optuna
final_model = build_model(**best_params).fit(X_train, y_train)
final_score = final_model.score(X_test, y_test)  # the ONE and ONLY test evaluation
```

## Putting it all together — a realistic end-to-end workflow

`best_practices_workflow.py` in this folder runs this exact progression on the Breast Cancer dataset:

1. Baseline with default hyperparameters.
2. Broad `RandomizedSearchCV` (wide ranges, cheap, cross-validated).
3. Narrowed Optuna study (informed by step 2's best region, more trials).
4. Final model retrained on full training data with the best hyperparameters.
5. **One** evaluation on the held-out test set.

## Additional practical tips

- **Set a random seed everywhere** (`random_state`) for reproducibility — otherwise you can't tell if a score difference is real or just noise from a different random split/initialization.
- **Log every trial** (Optuna does this automatically via `study.trials_dataframe()`) — you'll often want to revisit which hyperparameters mattered, not just the final winner.
- **Watch CV variance, not just the mean** — a hyperparameter combination with a slightly lower mean but much lower variance across folds is often the safer, more production-ready choice.
- **Don't over-tune on a small dataset** — with limited data, extensive tuning risks overfitting to the *validation folds themselves*; a simpler model with light tuning can generalize better than a heavily-tuned complex one.
- **Re-tune when your data changes meaningfully** — hyperparameters tuned for last year's data distribution aren't guaranteed to be optimal after a meaningful shift in the underlying data.

## Try it yourself
Run `best_practices_workflow.py` end to end and compare the CV score after each stage (baseline → Random Search → Optuna) — you should see a steady improvement, with diminishing returns at each step, which is the expected and healthy pattern.
