# Linear Regression —  (Experience vs Salary)

This uses the classic **"Years of Experience vs Salary"** dataset — a widely used real-world teaching dataset for simple linear regression (commonly seen in ML courses, e.g., on Kaggle).

## 1. The Real Dataset

| Employee | Years of Experience ($x$) | Salary in \$1000s ($y$) |
|----------|---------------------------|---------------------------|
| $E_1$ | 1.1 | 39.3 |
| $E_2$ | 1.3 | 46.2 |
| $E_3$ | 2.0 | 43.5 |
| $E_4$ | 3.2 | 64.4 |
| $E_5$ | 4.0 | 57.0 |
| $E_6$ | 5.1 | 66.0 |
| $E_7$ | 6.0 | 93.9 |
| $E_8$ | 7.1 | 98.3 |
| $E_9$ | 8.2 | 113.8 |
| $E_{10}$ | 9.5 | 116.9 |

We want to fit a line:

$$
\hat{y} = w_1 x + w_0
$$

that best predicts Salary from Years of Experience, then predict for a new employee with $x = 6.5$ years.

## 2. Least Squares Formulas

$$
w_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

$$
w_0 = \bar{y} - w_1 \bar{x}
$$

## 3. Compute the Means

$$
\bar{x} = \frac{1.1+1.3+2.0+3.2+4.0+5.1+6.0+7.1+8.2+9.5}{10} = \frac{47.5}{10} = 4.75
$$

$$
\bar{y} = \frac{39.3+46.2+43.5+64.4+57.0+66.0+93.9+98.3+113.8+116.9}{10} = \frac{739.3}{10} = 73.93
$$

## 4. Compute Deviations and Products

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i-\bar{x})(y_i-\bar{y})$ | $(x_i-\bar{x})^2$ |
|------|------|------|------|------|------|
| 1.1 | 39.3 | -3.65 | -34.63 | 126.4 | 13.32 |
| 1.3 | 46.2 | -3.45 | -27.73 | 95.67 | 11.90 |
| 2.0 | 43.5 | -2.75 | -30.43 | 83.68 | 7.56 |
| 3.2 | 64.4 | -1.55 | -9.53 | 14.77 | 2.40 |
| 4.0 | 57.0 | -0.75 | -16.93 | 12.70 | 0.56 |
| 5.1 | 66.0 | 0.35 | -7.93 | -2.78 | 0.12 |
| 6.0 | 93.9 | 1.25 | 19.97 | 24.96 | 1.56 |
| 7.1 | 98.3 | 2.35 | 24.37 | 57.27 | 5.52 |
| 8.2 | 113.8 | 3.45 | 39.87 | 137.55 | 11.90 |
| 9.5 | 116.9 | 4.75 | 42.97 | 204.11 | 22.56 |

$$
\sum (x_i-\bar{x})(y_i-\bar{y}) = 754.33, \qquad \sum (x_i-\bar{x})^2 = 77.40
$$

## 5. Compute the Slope $w_1$

$$
w_1 = \frac{754.33}{77.40} \approx 9.75
$$

## 6. Compute the Intercept $w_0$

$$
w_0 = \bar{y} - w_1 \bar{x} = 73.93 - (9.75)(4.75) = 73.93 - 46.31 = 27.62
$$

## 7. The Fitted Regression Line

$$
\hat{y} = 9.75x + 27.62
$$

This means: **each additional year of experience is associated with about \$9,750 more salary**, and a hypothetical employee with 0 years of experience would start around \$27,620.

## 8. Predict for a New Employee

For $x = 6.5$ years:

$$
\hat{y} = 9.75(6.5) + 27.62 = 63.38 + 27.62 = 91.00
$$

$$
\hat{y}(6.5) \approx \$91{,}000
$$

## 9. Measuring Fit — Residuals

The residual for each point is:

$$
e_i = y_i - \hat{y}_i
$$

Example for $E_7 = (6.0, 93.9)$:

$$
\hat{y}_7 = 9.75(6.0) + 27.62 = 86.12
$$

$$
e_7 = 93.9 - 86.12 = 7.78
$$

## 10. Sum of Squared Errors (SSE)

$$
SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

This is exactly the quantity least squares minimizes — the fitted line above is the one that makes $SSE$ as small as possible for this data.

## 11. Coefficient of Determination ($R^2$)

$$
R^2 = 1 - \frac{SSE}{SST}, \qquad SST = \sum_{i=1}^{n}(y_i - \bar{y})^2
$$

$R^2$ tells us the proportion of variance in Salary explained by Experience. For this real dataset, simple linear regression on Experience vs Salary typically achieves:

$$
R^2 \approx 0.96
$$

meaning experience explains about 96% of the variation in salary in this dataset — a very strong linear relationship.

## 12. General Prediction Formula

$$
\hat{y}(x) = w_1 x + w_0
$$

## 13. Multiple Linear Regression (Generalization)

With more than one feature $x_1, x_2, \dots, x_p$ (e.g., adding Education Level, Age), the model generalizes to:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p = w_0 + \sum_{j=1}^{p} w_j x_j
$$

solved in matrix form using the **normal equation**:

$$
w = (X^T X)^{-1} X^T y
$$

where $X$ is the design matrix of features (with a column of 1s for the intercept) and $y$ is the vector of targets.
