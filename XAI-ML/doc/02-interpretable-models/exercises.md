# Exercises — 02 Interpretable Models

1. **Run it.** Open `code.ipynb`, run it end-to-end, and record the decision tree's accuracy and the logistic regression's top 3 features by absolute coefficient.
2. **Depth vs interpretability.** Retrain the decision tree with `max_depth=1`, then `max_depth=6`. What happens to accuracy? At what depth does the tree stop being easily readable by a human?
3. **Sign interpretation.** For the top 3 logistic regression coefficients you found, explain in plain English what a positive vs a negative coefficient means for the prediction (recall class 1 = Benign here).
4. **Compare feature rankings.** Are the top features from the decision tree's `feature_importances_` the same as the top features from logistic regression's coefficients? If not, why might two "interpretable" models disagree on what matters most?
5. **Extend.** Standardize the features (`StandardScaler`) before fitting logistic regression, and refit. Do the top features by |coefficient| change? Why does scaling matter for coefficient-based interpretation but not for decision trees?
