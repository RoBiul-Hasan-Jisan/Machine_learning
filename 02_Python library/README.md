# Python ML Libraries



Hands-on notes and exercises for the core Python data-science and machine-learning
stack — NumPy, pandas, Matplotlib, PyTorch, and scikit-learn — organized one library
per folder. This is a companion to
[python-for-machine-learning](https://github.com/RoBiul-Hasan-Jisan/Machine_learning/tree/main/Python%20For%20Machine%20Learning),
which covers core Python itself.

## Repository structure

```
python-ml-libraries/
├── numpy/                  # Arrays, vectorization, array attributes
├── pandas/                 # DataFrames, operations, an applied churn analysis
├── matplotlib/             # Plot types, a data-viz cheat sheet, a dashboard script
├── pytorch/                # Tensors and core PyTorch building blocks
├── scikit-learn/           # An end-to-end modeling workflow
├── lectures/               # Reference lecture slides (PDF)               
└── README.md
```

Each library folder has its own `README.md` describing the notebooks inside and any
data files they depend on.

## Contents at a glance

| Library | What's covered |
|---------|-----------------|
| [`numpy/`](numpy/) | Array creation, indexing, broadcasting, vectorization vs. Python loops, array attributes |
| [`pandas/`](pandas/) | DataFrame basics, `groupby`, cleaning/aggregation, and a customer-churn analysis exercise |
| [`matplotlib/`](matplotlib/) | Core plot types, a data-visualization cheat sheet, and an advanced multi-panel weather dashboard |
| [`pytorch/`](pytorch/) | Tensor basics and the core PyTorch workflow |
| [`scikit-learn/`](scikit-learn/) | A complete classification workflow — split, train, cross-validate, evaluate — on the Iris dataset |

## Getting started

**Prerequisites:** Python 3.10+ and Jupyter.

```bash
# Clone the repository
git clone https://github.com/<your-username>/python-ml-libraries.git
cd python-ml-libraries

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Open any notebook and run its cells in order. Notebooks that read a local CSV expect
to be run from inside their own folder (paths are relative — see each folder's
README for details).

## Note on data dependencies

`pandas/02_churn_analysis.ipynb` expects a `telecom_churn.csv` file that was not
included in this repo. Add your own copy to `pandas/data/` (or point the notebook
at a dataset of your choosing) before running it.

