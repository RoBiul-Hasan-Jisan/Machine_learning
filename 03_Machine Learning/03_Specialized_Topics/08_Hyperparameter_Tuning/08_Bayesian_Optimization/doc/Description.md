#  Bayesian Optimization

## Basic idea

Grid Search and Random Search both choose *where to look next* without learning anything from past trials — every combination is picked blind (systematically for Grid, randomly for Random). Bayesian Optimization is smarter: it builds a **probabilistic model** (commonly a Gaussian Process) of "hyperparameters → performance" based on every trial so far, and uses that model to pick the *most promising next point to try* — balancing:

- **Exploitation** — trying points near where performance has been good so far.
- **Exploration** — trying points in regions we're still uncertain about (which might hide an even better spot).

This exploit/explore balance is typically captured by an **acquisition function** (e.g., Expected Improvement) computed from the surrogate model's predicted mean and uncertainty at each candidate point.

```
1. Try a few random hyperparameter combinations, record their scores.
2. Fit a surrogate model (e.g., Gaussian Process) on (hyperparameters -> score) pairs so far.
3. Use the surrogate model to predict where the next BEST hyperparameter combination likely is
   (balancing "looks promising" vs "we're uncertain here, worth checking").
4. Actually try that combination, record its real score.
5. Repeat steps 2-4 for a fixed budget of trials.
6. Return the best combination found.
```

## Advantages over Grid Search

| | Grid / Random Search | Bayesian Optimization |
|---|---|---|
| Uses past trial results? | No — every trial is independent | Yes — each trial informs the next |
| Sample efficiency | Needs many trials to cover a space well | Often finds strong regions in far fewer trials |
| Best for | Cheap-to-train models, small spaces | Expensive-to-train models (deep nets, large boosted trees) where every trial is costly |
| Complexity to implement | Trivial | More complex — needs a surrogate model + acquisition function (usually via a library) |

**The practical tradeoff:** Bayesian Optimization pays off most when each individual model training run is expensive — if training takes seconds, the overhead of fitting a surrogate model isn't worth it and Random Search is simpler and fast enough. If training takes many minutes or hours (e.g., large gradient boosting models, deep learning), the sample efficiency of Bayesian Optimization can save enormous amounts of compute.

## A minimal from-scratch illustration

`bayesian_optimization_demo.py` in this folder implements a tiny Bayesian Optimization loop **from scratch** using `sklearn`'s `GaussianProcessRegressor` as the surrogate model, tuning a single hyperparameter of a Random Forest — purely to make the *mechanism* visible, not as a production tool.

## In practice: use a library
Implementing Bayesian Optimization from scratch (as the demo does) is purely educational. In real projects, use a dedicated library — **Optuna** (folder 09) is the most popular modern choice, using a related but distinct sampling strategy (Tree-structured Parzen Estimator, TPE) that scales better to many hyperparameters than classic Gaussian-Process-based Bayesian Optimization.

## Try it yourself
Run `bayesian_optimization_demo.py` and watch it print each trial's chosen hyperparameter and score — compare how quickly it converges toward a good region versus how many random trials it would take to stumble onto the same area.
