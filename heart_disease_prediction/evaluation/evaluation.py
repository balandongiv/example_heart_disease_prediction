from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Trains a given model on the training data and evaluates its performance on the test data.
    Returns a dictionary containing various evaluation metrics.

    Parameters:
    model: The machine learning model to be trained and evaluated.
    x_train: Training feature set.
    y_train: Training labels.
    x_test: Testing feature set.
    y_test: True labels for the test set.

    Returns:
    dict: A dictionary containing Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_hat = np.where(y_pred >= 0.5, 1, 0).flatten()

    metrics = {
        "Accuracy": accuracy_score(y_test, y_hat),
        "Precision": precision_score(y_test, y_hat, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_hat, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_hat, average='weighted', zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_hat)
    }

    return metrics
