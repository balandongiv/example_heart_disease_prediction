import os
import requests
import pandas as pd
from heart_disease_prediction.config import DATA_PATH, DOWNLOAD_PATH

# Kaggle dataset file URL (Direct Download)
KAGGLE_DATA_URL = "https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset/download?select=heart_disease_risk_dataset_earlymed.xls"

def download_dataset():
    """
    Downloads the dataset from the Kaggle URL and saves it in the specified directory.
    """
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    print(f"⚠️ Dataset not found at {DATA_PATH}. Downloading from Kaggle...")

    try:
        response = requests.get(KAGGLE_DATA_URL, stream=True)
        if response.status_code == 200:
            with open(DATA_PATH, "wb") as file:
                file.write(response.content)
            print(f"✅ Dataset downloaded successfully to {DATA_PATH}")
        else:
            print(f"❌ Failed to download dataset. HTTP Status Code: {response.status_code}")
            print("👉 Please manually download the dataset from Kaggle and place it in the specified directory.")

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("👉 Please manually download the dataset from Kaggle and place it in the specified directory.")

def check_and_download_data():
    """
    Checks if the dataset exists locally. If not, downloads it directly.
    """
    if os.path.exists(DATA_PATH):
        print(f"✅ Dataset found at {DATA_PATH}.")
    else:
        download_dataset()

def load_data():
    """
    Loads the heart disease dataset after checking for its existence.
    """
    check_and_download_data()

    if os.path.exists(DATA_PATH):
        try:
            # Specify Excel engine explicitly
            data = pd.read_csv(DATA_PATH)
            print("✅ Data loaded successfully.")
            return data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            exit()
    else:
        print(f"❌ Dataset file {DATA_PATH} not found even after attempted download.")
        exit()
