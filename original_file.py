
import kagglehub
mahatiratusher_heart_disease_risk_prediction_dataset_path = kagglehub.dataset_download('mahatiratusher/heart-disease-risk-prediction-dataset')

print('Data source import complete.')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""Dataset: https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset/data"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Sequential, callbacks, regularizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import sigmoid

"""# Data"""

data = pd.read_csv('/kaggle/input/heart-disease-risk-prediction-dataset/heart_disease_risk_dataset_earlymed.csv')
data.head()

# Check for the missing values
data.isnull().sum()

#assigning intup (x) and output (y: person has Heart Risk or not)
x = data.drop("Heart_Risk", axis=1)
y = data["Heart_Risk"]

# splitting data to train, val and test sets
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.20, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

print("x_train:", x_train.shape[0])
print("x_val:", x_val.shape[0])
print("x_test:", x_test.shape[0])

"""# Machine Learning Models: Logistic Regression, Decision Tree, Random Forest"""

model_list = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate our models
def evaluate (model, x_train, y_train, x_test, y_test):

    model.fit(x_train, y_train) # Train our models

    y_pred = model.predict(x_test) # Make predictions
    y_hat = np.where(y_pred >= 0.5, 1, 0).flatten() # Converting predictions to yes (1) or no (0)

    accuracy = accuracy_score(y_test, y_hat) # Evaluate accuracy
    precision = precision_score(y_test, y_hat, average='weighted', zero_division=0) # Evaluate precision
    recall = recall_score(y_test, y_hat, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_hat, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_hat)

    return accuracy, precision, recall, f1, cm

# Show results in a DataFrame
results = []

for name in model_list:
    accuracy, precision, recall, f1, cm = evaluate(model_list[name], x_train, y_train, x_test, y_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    })
results = pd.DataFrame(results).set_index("Model")
results

# Plot confusion matrices for our models

for name in model_list:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix - {name}")
plt.show()

"""Based on accuracy, precision, recall and f1 score **Logistic Regression** model is the best. I doubt we can improve these amazing results with deep learning, but out of interest we can see what neural network will predict :)

# Deep Learning (Neural Network)
"""

model = Sequential(
    [
        Dense(64, activation = 'gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(64, activation = 'gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(32, activation = 'gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(16, activation = 'gelu', kernel_regularizer=regularizers.l2(0.0001)),
        Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.0001))
    ]
)

model.compile(loss = "binary_crossentropy",
              optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
              metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(

min_delta=0.1,
patience=20,
restore_best_weights=True)

model.fit(
     x_train,y_train,
     validation_data=(x_val, y_val),
     epochs=30,
     callbacks=[early_stopping],
     verbose=0)

print(f"Max validation accuracy: {(max(model.history.history['val_accuracy'])):.2f}")

#plot the model

fig, (plt_acc, plt_loss) = plt.subplots(1, 2, figsize=(10, 5))

history = model.history.history
plt_acc.plot(history['accuracy'], label="Training")
plt_acc.plot(history['val_accuracy'], label="Validation")

plt_loss.plot(history['loss'], label="Training")
plt_loss.plot(history['val_loss'], label="Validation")

plt_acc.set_title("Accuracy")
plt_loss.set_title("Loss")
plt_acc.legend()
plt_loss.legend()

#We can tune the threshold. Here, for example, we will go for the best F1 score in the validation set:
ypred_val = model.predict(x_val)

best_threshold = 0
best_F1 = 0

step_size = (np.max(ypred_val) - np.min(ypred_val)) / 10000

for threshold in np.arange(np.min(ypred_val), np.max(ypred_val), step_size):

    yhat_val = ypred_val>=threshold
    F1 = f1_score(y_val, yhat_val, average='weighted', zero_division=0)

    if F1 > best_F1:
        best_F1 = F1
        best_threshold = threshold

print(f"best_threshold: {best_threshold:.2f}")

ypred_test = model.predict(x_test)
yhat_test = np.where(ypred_test >= best_threshold, 1, 0).flatten()
error_test = np.mean(yhat_test != y_test) *100
print(f"Test_Error: {error_test:.2f}%")

print(f"Accuracy on the test set: {(accuracy_score(y_test, yhat_test)):.4f}")
print(f"Precision: {(precision_score(y_test, yhat_test, average='weighted', zero_division=0)):.4f}")
print(f"Recall: {(recall_score(y_test, yhat_test, average='weighted', zero_division=0)):.4f}")
print(f"F1: {(f1_score(y_test, yhat_test, average='weighted', zero_division=0)):.4f}")

cm = confusion_matrix(y_test, yhat_test)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - TensorFlow model')
plt.show()

"""Unexpectedly (at least for me), our **TensorFlow model** gives slightly better results on the confusion matrix (54 missclassifications) than logistic regression (59 missclassifications)."""

