
# 2. Underfitting in Machine Learning

## What is Underfitting?

Underfitting occurs when a machine learning model is **too simple** to learn the important patterns and relationships present in the data.

The model fails to capture the underlying structure of the problem, resulting in poor performance on both the training data and new unseen data.

In simple terms:

> **The model has not learned enough from the data and makes overly simple assumptions.**

A model that underfits does not have enough learning capacity to understand the complexity of the problem.

---

### Example: House Price Prediction

Imagine building a model to predict house prices.

### Real relationship:


**House Price =Location + Size + Number of Rooms + Age + Neighborhood + Market Conditions**


A good model learns these relationships.

---

### Underfitted Model:

The model only uses one simple rule:


Price = House Size × Fixed Rate


It ignores important factors:

- Location
- Number of bedrooms
- Nearby facilities
- Market trends

Prediction:

```bash
Small house in a premium city area
↓
Model predicts low price

Large house in a poor location
↓
Model predicts high price

```
---
The model fails because it is too simple to understand the real-world complexity.

---

## Signs of Underfitting

A model is likely underfitting when:


- Training Performance: Poor
- Validation/Test Performance: Poor


Example:

| Dataset | Accuracy |
|---|---:|
| Training Data | 65% |
| Validation Data | 63% |
| Test Data | 62% |

The model performs poorly everywhere because it has not learned enough patterns.

---

# Primary Causes of Underfitting

## 1. Model Is Too Simple

A model may not have enough capacity to represent complex relationships.

## Example:
```bash
Using linear regression for a highly nonlinear problem:


Linear Model:

y = ax + b


But the real relationship is:


y = complex nonlinear function

```
---
The model cannot capture the true pattern.

---

## 2. Insufficient Training

The model may not have enough time to learn.

Common causes:

- Too few training epochs
- Training stopped too early
- Poor optimization settings

### Example:

A neural network trained for:

```bach
5 epochs → Not enough learning

100 epochs → Better pattern understanding
```

---

## 3. Poor Feature Selection

Important information may be missing from the input data.

### Example:

> Predicting student performance:

**Useful features:**


- Study hours
- Attendance
- Previous grades
- Assignment scores


**If the model only receives:**


- Student ID
- Age


> it lacks the information needed to make accurate predictions.

---

## 4. Poor Feature Engineering

Raw data may not contain useful representations.

### Example:

*Raw feature:*

Date:2026-06-25


**Better features:**


- Day of week
- Month
- Holiday indicator
- Season



Better features help the model discover important patterns.

---

## 5. Excessive Regularization

Regularization prevents overfitting, but too much regularization can make the model too simple.

### Example:

*Too much penalty:*

```bash
Model complexity ↓
Learning ability ↓
Underfitting ↑
```


---

# How to Reduce Underfitting

## 1. Increase Model Complexity

Use a more powerful model.

### Examples:

- Add more neural network layers
- Increase model capacity
- Use nonlinear algorithms

### Example:

*Replace:*


- Linear Regression


**with:**


- Random Forest
- Gradient Boosting
- Neural Network


---

## 2. Train for Longer

Allow the model more time to learn.

Increase:

- Number of epochs
- Training iterations

Monitor validation performance to avoid moving from underfitting into overfitting.

---

## 3. Improve Feature Engineering

Create better input features.

### Example:

**For predicting customer purchases:**

> Instead of:


- Customer ID


**Use:**


- Purchase frequency
- Average spending
- Previous purchases
- Customer activity


---

## 4. Collect More Relevant Data

More high-quality examples can help the model discover important relationships.

### Example:

> Instead of:


1,000 customer records


*Use:*


1,000,000 diverse customer records


---

## 5. Reduce Excessive Regularization

If the model is too restricted:

**Reduce:**

- L1 penalty
- L2 penalty
- Dropout rate

Allow the model to learn more complex patterns.

---

## 6. Use Better Algorithms

Some algorithms naturally handle complex patterns better.

### Examples:

- Random Forest
- Gradient Boosting
- Deep Neural Networks

---

## 7. Hyperparameter Tuning

Experiment with different settings:

### Examples:

- Learning rate
- Model size
- Number of layers
- Batch size
- Regularization strength

> The goal is to find the right balance between simplicity and complexity.

---

## How to Detect Overfitting and Underfitting

## Detecting Overfitting

### 1. Compare Training and Validation Performance

> Overfitting pattern:


- Training Performance: Very High

- Validation Performance: Much Lower


###  Example:


- Training Accuracy: 99%

- Validation Accuracy: 75%


The model memorized training data.

---

### 2. Learning Curves

Plot training and validation performance over time.

> Overfitting:


- Training Error ↓↓↓

- Validation Error ↓ then ↑


The model improves on training data but gets worse on unseen data.

---

### 3. Cross-Validation

Train and evaluate the model on multiple data splits.

Overfitting models often show unstable performance across different folds.

---

### 4. Regularization Testing

Increase regularization strength.

If validation performance improves, the original model was likely overfitting.

---

## Detecting Underfitting

## 1. Training and Validation Performance

Underfitting pattern:


- Training Performance: Low

- Validation Performance: Low


### Example:


- Training Accuracy: 65%

- Validation Accuracy: 63%


The model has not learned enough.

---

## 2. Learning Curves

Underfitting:


- Training Error: High

- Validation Error: High

Both curves remain poor


More training data does not significantly improve performance.

---

## 3. Feature Analysis

Check whether important information is missing.

Questions:

- Are important features included?
- Are features meaningful?
- Does the model have enough information?

---

## 4. Model Complexity Analysis

Ask:

- Is the algorithm powerful enough?
- Does it capture nonlinear relationships?
- Are there too many restrictions?

---

## Overfitting vs Underfitting Summary

| | Underfitting | Good Fit | Overfitting |
|-|-|-|-|
| Model Complexity | Too simple | Appropriate | Too complex |
| Bias | High | Balanced | Low |
| Variance | Low | Balanced | High |
| Training Error | High | Low | Very Low |
| Test Error | High | Low | High |
| Learns | Too little | General patterns | Noise |

---

## Key Idea

The goal of machine learning is to find the right balance:

- **Avoid underfitting:** The model must be powerful enough to learn important patterns.
- **Avoid overfitting:** The model must not memorize noise.

A successful model learns the true structure of the data and generalizes well to examples it has never seen.

# Balancing Overfitting and Underfitting

Building a successful machine learning model requires finding the right balance between **underfitting** and **overfitting**.

A model should be complex enough to learn meaningful patterns from data, but not so complex that it memorizes noise and fails on new examples.

The ultimate goal of machine learning is not to achieve perfect performance on training data. The goal is to build a model that **generalizes well to unseen data**.

---
