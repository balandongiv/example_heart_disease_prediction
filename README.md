# About the Project

This project aims to migrate from a notebook-based workflow to a modularized Python codebase using `.py` files.

The original code was obtained from the following Kaggle dataset:
[Heart Disease Risk Prediction Dataset](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset?select=heart_disease_risk_dataset_earlymed.csv)

# Downloading the Original Notebook Code

The original Jupyter Notebook code can be downloaded from the [Heart Disease Risk Prediction Dataset](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset?select=heart_disease_risk_dataset_earlymed.csv) Kaggle page. To download the `.ipynb` file, navigate to the notebook section (often indicated by a code icon or tab), find the desired notebook, and typically there will be a download button (often represented by three vertical dots or a download icon). Click this button to download the notebook.

# Downloading the CSV Data

Initially, the intention was to programmatically download the CSV dataset. However, due to time constraints and after several attempts, the decision was made to manually download the file.  Please download the `heart_disease_risk_dataset_earlymed.csv` file from the **Datasets** tab on the Kaggle dataset page linked above and place it in the `data` folder.

# Decomposing the Notebook Code into Modules

When decomposing the original notebook (`.ipynb`) into modular Python files (`.py`), please adhere to the folder structure outlined below. This structure organizes the code into logical components: `data`, `preprocessing`, `models`, `evaluation`, `utils`, and `tests`.

While there are no strict naming conventions for the `.py` files within each folder, consider adopting descriptive names for clarity and maintainability.  For example, you might use the following naming scheme:

- `data/data_loader.py`
- `preprocessing/preprocessing.py`
- `models/ml_models.py`
- `models/deep_learning.py`
- `evaluation/evaluation.py`
- `utils/visualization.py`
- `tests/test_data_loader.py` (and similar `test_*.py` for other modules)
```
heart_disease_prediction/
├── main.py
├── config.py
├── requirements.txt
│
├── data/
│   └── data_loader.py
│
├── preprocessing/
│   └── preprocessing.py
│
├── models/
│   ├── ml_models.py
│   └── deep_learning.py
│
├── evaluation/
│   └── evaluation.py
│
├── utils/
│   └── visualization.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_ml_models.py
│   ├── test_deep_learning.py
│   └── test_evaluation.py
```
