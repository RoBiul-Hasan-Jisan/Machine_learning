# pandas

DataFrame fundamentals plus an applied exercise using pandas for exploratory
analysis.

| Notebook | Description | Data used |
|----------|-------------|-----------|
| [01_basic_operations.ipynb](01_basic_operations.ipynb) | Creating DataFrames, indexing, filtering, and basic operations. | `data/data.csv`, `data/cleaned_sales_data.csv` |
| [02_churn_analysis.ipynb](02_churn_analysis.ipynb) | Exploratory analysis of customer churn (e.g., average daytime phone usage by churned users). | `telecom_churn.csv` *(not included — see note below)* |

## Data

- `data/data.csv` — small sample dataset (name, age, city).
- `data/cleaned_sales_data.csv` — sample sales data with revenue, expenses, and a
  computed moving average, used for aggregation/cleaning examples.

`02_churn_analysis.ipynb` expects a `telecom_churn.csv` file that wasn't part of
the original export. Add your own copy to `data/` (or swap in a similar dataset)
before running that notebook.
