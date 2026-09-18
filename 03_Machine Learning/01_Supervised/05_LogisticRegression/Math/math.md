# Logistic Regression —  (Hours Studied vs Exam Pass/Fail)

This uses the well-known **"Hours Studied vs Passing an Exam"** dataset (a real, widely-cited example used to introduce logistic regression, popularized in the Wikipedia logistic regression article).

## 1. The Real Dataset

20 students, each recorded by hours studied and whether they passed ($1$) or failed ($0$):

| Hours ($x$) | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 1.75 | 2.00 | 2.25 | 2.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Pass ($y$) | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |

| Hours ($x$) | 2.75 | 3.00 | 3.25 | 3.50 | 4.00 | 4.25 | 4.50 | 4.75 | 5.00 | 5.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Pass ($y$) | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |

We want to predict the probability of passing given hours studied, then classify a new student who studied $x = 3.0$ hours (note: an actual student in the data who studied exactly 3.0 hours failed — a useful real edge case).

## 2. Why Not Linear Regression Here?

Linear regression can predict values outside $[0, 1]$, which makes no sense for a probability. Logistic regression instead models the **log-odds** as linear in $x$, and squashes the output into $[0,1]$ using the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

## 3. The Model

$$
z = w_1 x + w_0
$$

$$
P(y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-(w_1 x + w_0)}}
$$

## 4. Fitting via Maximum Likelihood

Unlike linear regression, there's no closed-form solution — $w_0, w_1$ are found by maximizing the **log-likelihood**:

$$
\ell(w) = \sum_{i=1}^{n} \Big[ y_i \log P(y_i=1\mid x_i) + (1-y_i)\log\big(1 - P(y_i=1\mid x_i)\big) \Big]
$$

equivalently, minimizing the **binary cross-entropy loss**:

$$
J(w) = -\frac{1}{n}\sum_{i=1}^{n} \Big[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \Big]
$$

solved iteratively via **gradient descent**, using the gradient:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{n}\sum_{i=1}^{n} (\hat{p}_i - y_i)\, x_{ij}
$$

with the update rule:

$$
w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j}
$$

where $\alpha$ is the learning rate.

## 5. Fitted Coefficients (Real Result for This Dataset)

Running maximum-likelihood estimation on this exact real dataset converges to approximately:

$$
w_0 \approx -4.0777, \qquad w_1 \approx 1.5046
$$

So the fitted model is:

$$
z = 1.5046x - 4.0777
$$

$$
P(\text{Pass} \mid x) = \frac{1}{1 + e^{-(1.5046x - 4.0777)}}
$$

## 6. Interpreting the Coefficients

- $w_1 = 1.5046 > 0$: more hours studied increases the log-odds of passing.
- The **decision boundary** ($P = 0.5$, i.e., $z = 0$) occurs at:

$$
1.5046x - 4.0777 = 0 \ \Rightarrow \ x = \frac{4.0777}{1.5046} \approx 2.71 \text{ hours}
$$

Students studying more than ~2.71 hours are predicted more likely to pass than fail.

## 7. Predict for $x = 3.0$ Hours

$$
z = 1.5046(3.0) - 4.0777 = 4.5138 - 4.0777 = 0.4361
$$

$$
P(\text{Pass}) = \sigma(0.4361) = \frac{1}{1+e^{-0.4361}} = \frac{1}{1+0.6465} = 0.6072
$$

$$
P(\text{Pass} \mid x=3.0) \approx 0.607 \ (60.7\%)
$$

## 8. Classification Rule

$$
\hat{y}(x) = \begin{cases} 1 \ (\text{Pass}) & \text{if } P(y=1\mid x) \geq 0.5 \\ 0 \ (\text{Fail}) & \text{if } P(y=1\mid x) < 0.5 \end{cases}
$$

Since $0.607 \geq 0.5$:

$$
\hat{y}(3.0) = \text{Pass}
$$

**Real-world note:** the actual student who studied 3.0 hours in this dataset failed — a genuine misclassification. This is expected and normal: logistic regression outputs *probabilities*, not certainties, and $60.7\%$ still leaves a real chance of failing. This is exactly the kind of boundary case real models get wrong.

## 9. Predict for $x = 5.0$ Hours (Clearer Case)

$$
z = 1.5046(5.0) - 4.0777 = 7.523 - 4.0777 = 3.4453
$$

$$
P(\text{Pass}) = \sigma(3.4453) = \frac{1}{1+e^{-3.4453}} \approx \frac{1}{1+0.0319} \approx 0.969
$$

$$
\hat{y}(5.0) = \text{Pass} \ (96.9\% \text{ confidence}) \quad \checkmark \text{ matches actual outcome}
$$

## 10. General Prediction Formula

$$
\hat{y}(x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}
$$

## 11. Multi-Feature Logistic Regression (Generalization)

With multiple features $x_1, \dots, x_p$ (e.g., adding hours slept, attendance rate):

$$
z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p
$$

$$
P(y=1 \mid x) = \frac{1}{1 + e^{-z}}
$$

The same sigmoid squashing and cross-entropy loss apply regardless of how many features are used.