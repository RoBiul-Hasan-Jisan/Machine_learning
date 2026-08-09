# Explainable AI (XAI) — A Complete Learning Repo

A self-contained, topic-by-topic course on Explainable AI: theory (math, taxonomy, regulation) paired with runnable code (SHAP, LIME, PDP/ICE, counterfactuals, a full XAI dashboard) on a single consistent dataset, plus exercises with solutions for every lesson.

## Repo layout

```
doc/    → one subfolder per topic, each with describe.md (theory + lesson overview), exercises.md, solutions.md
code/   → one notebook per topic that has runnable code, named <topic>.ipynb
```

```
.
├── doc/
│   ├── 01-foundations/describe.md, exercises.md, solutions.md
│   ├── 02-interpretable-models/describe.md, exercises.md, solutions.md
│   ├── 03-shap/describe.md, exercises.md, solutions.md
│   ├── 04-lime/describe.md, exercises.md, solutions.md
│   ├── 05-model-agnostic-pdp-ice/describe.md, exercises.md, solutions.md
│   ├── 06-counterfactual-explanations/describe.md, exercises.md, solutions.md
│   ├── 07-causal-and-concept-explanations/describe.md, exercises.md, solutions.md
│   ├── 08-evaluation-metrics/describe.md, exercises.md, solutions.md
│   ├── 09-xai-dashboard-and-trust/describe.md, exercises.md, solutions.md
│   ├── 10-modalities-regulation-ethics/describe.md, exercises.md, solutions.md
│   └── 11-limitations-and-open-problems/describe.md, exercises.md, solutions.md
├── code/
│   ├── 02-interpretable-models.ipynb
│   ├── 03-shap.ipynb
│   ├── 04-lime.ipynb
│   ├── 05-model-agnostic-pdp-ice.ipynb
│   ├── 06-counterfactual-explanations.ipynb
│   └── 09-xai-dashboard-and-trust.ipynb
├── requirements.txt
└── README.md
```

Topics `01`, `07`, `08`, `10`, `11` are theory-only (no matching notebook in `code/`) — their `doc/.../describe.md` covers the concepts and their `exercises.md` is conceptual rather than code-based.

## Learning path

| # | Topic | Doc | Code |
|---|-------|-----|------|
| 01 | Foundations of XAI — black box problem, accuracy/interpretability trade-off, DARPA framework, taxonomy | [`doc/01-foundations`](./doc/01-foundations) | — |
| 02 | Interpretable ("glass-box") models — Decision Trees, Logistic Regression | [`doc/02-interpretable-models`](./doc/02-interpretable-models) | [`code/02-interpretable-models.ipynb`](./code/02-interpretable-models.ipynb) |
| 03 | SHAP — Shapley values, axioms, TreeSHAP, global/local explanations | [`doc/03-shap`](./doc/03-shap) | [`code/03-shap.ipynb`](./code/03-shap.ipynb) |
| 04 | LIME — local surrogate models, sampling, kernel width | [`doc/04-lime`](./doc/04-lime) | [`code/04-lime.ipynb`](./code/04-lime.ipynb) |
| 05 | PDP & ICE — global, model-agnostic feature effect plots | [`doc/05-model-agnostic-pdp-ice`](./doc/05-model-agnostic-pdp-ice) | [`code/05-model-agnostic-pdp-ice.ipynb`](./code/05-model-agnostic-pdp-ice.ipynb) |
| 06 | Counterfactual explanations — minimal change to flip a prediction | [`doc/06-counterfactual-explanations`](./doc/06-counterfactual-explanations) | [`code/06-counterfactual-explanations.ipynb`](./code/06-counterfactual-explanations.ipynb) |
| 07 | Causal vs. associative explanations, SCMs, TCAV, uncertainty quantification | [`doc/07-causal-and-concept-explanations`](./doc/07-causal-and-concept-explanations) | — |
| 08 | Evaluation metrics — faithfulness, comprehensibility, fairness, robustness | [`doc/08-evaluation-metrics`](./doc/08-evaluation-metrics) | — |
| 09 | Capstone: `XAIDashboard` class + a composite trust score | [`doc/09-xai-dashboard-and-trust`](./doc/09-xai-dashboard-and-trust) | [`code/09-xai-dashboard-and-trust.ipynb`](./code/09-xai-dashboard-and-trust.ipynb) |
| 10 | XAI across modalities (tabular/image/text/time series); GDPR, FDA, Algorithmic Accountability Act | [`doc/10-modalities-regulation-ethics`](./doc/10-modalities-regulation-ethics) | — |
| 11 | Limitations, open research questions, method-selection checklist | [`doc/11-limitations-and-open-problems`](./doc/11-limitations-and-open-problems) | — |

**Suggested order:** follow the numbers top to bottom — each lesson builds on the last (foundations → glass-box models → post-hoc methods → evaluation → advanced/regulatory context).

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

All notebooks in `code/` use the same dataset — **Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`) — so every explanation method can be compared apples-to-apples on the same model and the same patients.

## What you'll be able to do by the end

- Explain *why* a black-box model made a specific prediction (SHAP, LIME)
- Explain *how* a model behaves in general (global feature importance, PDP/ICE)
- Tell a user *what to change* to get a different outcome (counterfactuals)
- Tell the difference between correlation-based and true causal explanations
- Quantify whether an explanation — and the model behind it — is actually trustworthy
- Know which method to reach for, for which data type, audience, and regulatory context
