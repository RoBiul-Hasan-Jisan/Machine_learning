# Exercises — 05 PDP & ICE

1. **Read the curve.** For `'mean area'`, describe in words what the PDP curve shows: as `mean area` increases, does the predicted probability of Benign go up or down, on average?
2. **Spot heterogeneity.** Look at the ICE lines under the PDP for `'mean area'`. Do all the individual lines follow the same trend as the average PDP line, or do some patients' curves diverge or even go the opposite direction?
3. **Add a feature.** Add a 4th feature (e.g. `'worst radius'`) to the `features` list in the PDP cell and regenerate the plot grid. Does its PDP shape look similar to `'mean area'`'s? Why might that be, given both relate to tumor size?
4. **Two-way PDP.** Using `PartialDependenceDisplay.from_estimator(rf, X_train, [('mean area', 'mean smoothness')], ...)`, generate a 2D PDP. What does the contour plot tell you about interaction between these two features that a 1D PDP for each separately would miss?
5. **Limitation.** PDP assumes features are independent when averaging out "other" features. Why is this a problem for the Breast Cancer dataset, where many features (e.g. `mean radius`, `mean perimeter`, `mean area`) are highly correlated with each other?
