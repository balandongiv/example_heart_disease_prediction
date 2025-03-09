
# About the Project

This project aims to migrate from a notebook-based workflow to a modularized Python codebase using `.py` files.

The original code was obtained from the following Kaggle dataset:
[Heart Disease Risk Prediction Dataset](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset?select=heart_disease_risk_dataset_earlymed.csv)

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

# Downloading the CSV Data

Initially, the intention was to programmatically download the CSV dataset. However, due to time constraints and after several attempts, the decision was made to manually download the file.  Please download the `heart_disease_risk_dataset_earlymed.csv` file from the **Datasets** tab on the Kaggle dataset page linked above and place it in the `data` folder.
