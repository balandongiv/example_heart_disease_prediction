import os

import os

# Path where the dataset should be downloaded
DOWNLOAD_PATH = r"C:\Users\balan\IdeaProjects\example_heart_disease_prediction"

# Full dataset file path (update to XLS extension)
DATA_PATH = os.path.join(DOWNLOAD_PATH, "heart_disease_risk_dataset_earlymed.xls")

DATA_PATH=r'C:\Users\balan\IdeaProjects\example_heart_disease_prediction\heart_disease_prediction\data\heart_disease_risk_dataset_earlymed.csv'
# DATA_PATH = "/kaggle/input/heart-disease-risk-prediction-dataset/heart_disease_risk_dataset_earlymed.csv"
TEST_SIZE = 0.2
VAL_SIZE = 0.5
RANDOM_STATE = 1
EPOCHS = 30
LEARNING_RATE = 0.001