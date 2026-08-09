# 01 · Foundations of Explainable AI (XAI)

## PART 1: FOUNDATIONS OF XAI

### 1.1 The Black Box Problem

**Definition**: A "black box" model is one where:
* We can observe inputs $\rightarrow$ outputs.
* But cannot understand internal decision logic.
* Parameters are too complex for human comprehension.

**Examples**:
* Deep Neural Networks (millions of parameters)
* Random Forests (hundreds of trees)
* Gradient Boosting ensembles

**Why black boxes emerged**:

| Feature | Interpretable Models (1990s) | Black Box Models (2010s+) |
| :--- | :--- | :--- |
| **Model Types** | Decision Trees, Linear/Logistic Regression, Rule-based systems | Deep Learning, Random Forests, XGBoost, Transformers |
| **Accuracy Gap** | 70-80% | 95-99% |
| **Interpretability** | High | Near zero |

### 1.2 The Trade-off: Accuracy vs Interpretability

**The Fundamental Law of XAI**:
* Interpretability $\propto$ 1/Complexity
* Accuracy $\propto$ Complexity (to a point)
* Optimal zone: High accuracy with sufficient interpretability

**Visual Representation**:
```bash
Accuracy
    ↑
1.0 |                    ●●●● (Deep Learning)
    |                 ●●
    |              ●●      ●●● (Random Forest)
0.8 |           ●●          ●
    |        ●●              ●● (Decision Tree)
0.6 |     ●●                  ●
    |  ●●                      ●● (Linear)
0.4 | ●                          ●
    |
    +--------------------------------→ Complexity
         Low              High
         ↑                         ↑
    Interpretable              Black Box

 ```
---
### Real-world example:

- Linear Regression: 75% accuracy, 100% interpretable

- Random Forest: 92% accuracy, 60% interpretable

- Deep Learning: 98% accuracy, 10% interpretable

## 1.3 The XAI Framework - DARPA's Model

DARPA (Defense Advanced Research Projects Agency) defined the standard XAI framework:
```bash
┌─────────────────────────────────────────────────────────┐
│                    TRAINING DATA                        │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────────────────────┐
│           Machine Learning Process                     │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │ Learned      │ →  │ Black Box    │                  │
│  │ Function     │    │ Model        │                  │
│  │ f(x) = y     │    │ (Complex)    │                  │
│  └──────────────┘    └──────┬───────┘                  │
└─────────────────────────────┼──────────────────────────┘
                            ↓
                    ┌──────────────────┐
                    │  XAI System      │
                    │  ┌────────────┐  │
                    │  │ Prediction │  │
                    │  │ Explanation│  │
                    │  │ Interface  │  │
                    │  └────────────┘  │
                    └────────┬─────────┘
                             ↓
              ┌──────────────────────────┐
              │   USER (Human)           │
              │   Understands WHY        │
              │   WHEN to trust          │
              │   HOW to improve         │
              └──────────────────────────┘


```

# Key Components:

- Prediction: What the model outputs

- Explanation: Why it made that prediction

- Uncertainty: How confident is the model

- Counterfactuals: What would change the outcome         



---

## PART 2: TAXONOMY OF XAI METHODS

### 2.1 By Timing (When explanation is generated)

```bash

XAI METHODS
│
├── ANTECEDENT (Before prediction)
│   ├── Glass-box models (inherently interpretable)
│   │   ├── Linear/Logistic Regression
│   │   ├── Decision Trees (shallow)
│   │   ├── Rule-based systems
│   │   └── K-Nearest Neighbors (K=small)
│   │
│   └── Design-time interpretability
│       ├── Feature selection
│       ├── Regularization (L1 for sparsity)
│       └── Model constraints
│
└── POST-HOC (After prediction)
    ├── Model-specific methods
    │   ├── Tree feature importance
    │   ├── Attention weights (Transformers)
    │   └── Gradient-based (Neural nets)
    │
    └── Model-agnostic methods
        ├── SHAP (Shapley values)
        ├── LIME (Local surrogates)
        ├── Counterfactuals
        └── PDP/ICE plots


```
---

### 2.2 By Scope (Local vs Global)

```bash

SCOPE DIMENSION
│
├── GLOBAL EXPLANATIONS (Entire model behavior)
│   │
│   ├── "How does the model work generally?"
│   ├── Examples:
│   │   ├── Feature importance (global)
│   │   ├── Partial Dependence Plots
│   │   ├── Model structure (tree visualization)
│   │   └── Surrogate models (global)
│   │
│   └── Use cases:
│       ├── Model validation
│       ├── Regulatory compliance
│       ├── Feature engineering
│       └── Debugging systematic biases
│
└── LOCAL EXPLANATIONS (Single prediction)
    │
    ├── "Why was THIS specific prediction made?"
    ├── Examples:
    │   ├── SHAP values (per instance)
    │   ├── LIME explanations
    │   ├── Counterfactuals
    │   └── Individual Conditional Expectation
    │
    └── Use cases:
        ├── High-stakes decisions (loan, medical)
        ├── User-facing explanations
        ├── Error analysis
        └── Right-to-explanation (GDPR)
```
---


**Example Comparison:**

### Global: "Age and income are top features overall"
Global_Importance = {"Age": 0.45, "Income": 0.32, "Debt": 0.23}

### Local: "For THIS person, high debt caused rejection"
Local_Explanation = {"Debt": -0.8, "Age": +0.1, "Income": +0.1}


## 2.3 By Method Type

``` bash
METHOD CATEGORIES
│
├── 1. FEATURE ATTRIBUTION
│   └── Assign importance scores to input features
│       ├── SHAP (Game theory-based)
│       ├── LIME (Local linear approximation)
│       ├── Integrated Gradients (Deep learning)
│       └── Occlusion/Removal (Simple perturbation)
│
├── 2. SURROGATE MODELS
│   └── Simple model that mimics complex one
│       ├── Global surrogates (e.g., rule lists)
│       ├── Local surrogates (LIME)
│       └── Tree surrogates (Distill model)
│
├── 3. EXAMPLE-BASED
│   └── Use data examples as explanations
│       ├── Counterfactuals ("Change X to get Y")
│       ├── Prototypes (Typical examples)
│       ├── Criticisms (Edge cases)
│       └── Influential instances (Training data impact)
│
├── 4. VISUALIZATION
│   └── Visual representation of internal states
│       ├── Saliency maps (Images)
│       ├── Attention visualization (NLP)
│       ├── Activation maximization
│       └── Concept Activation Vectors
│
└── 5. HYBRID METHODS
    └── Combine multiple approaches
        ├── SHAP + LIME (Validate consistency)
        ├── Counterfactual + Feature attribution
        └── Global + Local explanations

```
---

