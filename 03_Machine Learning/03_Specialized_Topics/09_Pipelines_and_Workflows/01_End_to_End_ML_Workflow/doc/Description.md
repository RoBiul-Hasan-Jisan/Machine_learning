# 01. End-to-End ML Workflow

## Learning Objectives

- Name and order the stages of a real ML project, from problem definition to monitoring
- Explain what can go wrong at each stage and why most ML failures are not modeling failures
- Understand how the rest of this module (02-11) maps onto this workflow

## The Problem

Tutorials teach `model.fit(X, y)`. Production ML is everything around that one line: where the data came from, whether it will still be there next month, whether your validation number means anything, and what happens when the model is wrong in front of a customer.

Most ML projects fail for reasons that have nothing to do with model choice: badly defined success metrics, leaked labels, silent data drift, or no plan for retraining. This lesson lays out the full pipeline so the rest of the module has a map to hang on.

## The Concept

```
Problem Definition
     ↓
Data Collection
     ↓
Data Cleaning
     ↓
EDA
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Evaluation
     ↓
Tuning
     ↓
Deployment
     ↓
Monitoring
```

This is drawn as a straight line, but it is really a loop. Monitoring feeds back into problem definition (is this still the right target?), EDA (has the input distribution shifted?), and training (time to retrain?). Treat the diagram as one lap of a cycle you will run many times.

### Problem Definition

Before touching data, answer:
- What decision does this prediction drive? If nobody acts differently based on the output, you don't need a model.
- What is the unit of prediction? (one row = one customer? one transaction? one day?)
- What metric reflects business success, and how does it map to an ML metric (accuracy, AUC, RMSE)? These are often not the same thing — a 1% accuracy gain can be worthless if the business cares about the top 50 highest-risk cases.
- What is the cost of a false positive vs a false negative? This should shape your metric choice and decision threshold, not just the model.

### Data Collection

Where does training data come from, and will that same data be available, in the same form, at prediction time? A classic failure: a feature is available historically (e.g. "did the customer call support after purchase") but does not exist yet at the moment you need to make the prediction. That is a **label leakage** trap disguised as a feature — covered in depth in Lesson 02.

### Data Cleaning

Handling missing values, fixing types, deduplicating rows, resolving inconsistent categories ("NY" vs "New York" vs "ny"). This is typically 60-80% of project time. Decisions made here (how to impute, whether to drop rows) must be reproducible at inference time, which is why Lessons 03-06 wrap cleaning into pipeline objects rather than one-off scripts.

### EDA (Exploratory Data Analysis)

Look at distributions, class balance, correlations, and outliers before modeling. EDA answers: is this problem even learnable from this data? Are there obvious leaks (a feature perfectly separating classes)? Is the target imbalanced enough to need resampling or a different metric?

### Feature Engineering

Transforming raw columns into model-ready inputs: scaling, encoding, interaction terms, aggregations, domain-specific features. Covered in depth in Lesson 06.

### Model Training

Fitting one or more candidate models. This is the step tutorials focus on, and it is usually the fastest step in a real project once the pipeline exists.

### Evaluation

Measuring performance on held-out data using the metric chosen in Problem Definition, not just whatever `.score()` returns by default. Includes checking performance across important subgroups, not just in aggregate.

### Tuning

Hyperparameter search (grid search, random search, Bayesian optimization), always wrapped in proper cross-validation (Lesson 08) so tuning decisions don't leak into your evaluation number.

### Deployment

Packaging the trained model (and every preprocessing step it depends on) so it can produce predictions on new data. Covered in Lessons 09-10.

### Monitoring

Tracking prediction distributions, input feature drift, and — when ground truth eventually arrives — actual model performance in production. A model that scored well at training time silently degrades as the world changes; monitoring is how you find out before your stakeholders do.

## How this module is organized

| Lesson | Workflow stage(s) it covers |
|---|---|
| 02 Train/Validation/Test Split | Evaluation methodology, leakage prevention |
| 03 Sklearn Pipeline | Chaining preprocessing + model into one object |
| 04 ColumnTransformer | Feature Engineering (mixed column types) |
| 05 Custom Transformers | Feature Engineering (domain-specific logic) |
| 06 Feature Engineering Pipeline | Feature Engineering (full pattern) |
| 07 Model Training Pipeline | Model Training, experiment tracking |
| 08 Cross-Validation Pipeline | Evaluation, Tuning |
| 09 Model Persistence | Deployment (saving artifacts) |
| 10 Inference Pipeline | Deployment (serving predictions) |
| 11 ML Project Structure | All stages, organized into a real repo |

## Exercises

1. Pick a dataset you know well (or `sklearn.datasets`). Write one sentence each for Problem Definition, the ML metric, and the business metric. Are they the same thing? If not, explain the gap.
2. List three features in that dataset that might not be available at true prediction time (potential leakage). Justify each.
3. Sketch what "monitoring" would concretely mean for that dataset in production — what would you log, and what would trigger a retrain?

## Key Terms

| Term | What it actually means |
|---|---|
| ML workflow | The repeatable sequence from problem definition through monitoring; a loop, not a line |
| Business metric vs ML metric | The metric stakeholders care about vs the metric optimized during training; they must be connected deliberately |
| Data drift | Change in the input feature distribution over time, which can silently degrade a deployed model |
| Label leakage | A feature that encodes information unavailable (or different) at true prediction time |
| Retraining trigger | A defined condition (performance drop, drift threshold, time elapsed) that prompts retraining a deployed model |
