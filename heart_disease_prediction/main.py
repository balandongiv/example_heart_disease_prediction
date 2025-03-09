from data.data_loader import load_data
from evaluation.evaluation import evaluate_model
from models.ml_models import get_models
from preprocessing.preprocessing import preprocess_data
from utils.visualization import plot_confusion_matrix


def train_and_evaluate_ml_models(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates machine learning models.

    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Testing features
        y_test: Testing labels
    """
    print("Training machine learning models...")
    models = get_models()  # Retrieve dictionary of ML models

    for name, model in models.items():
        metrics = evaluate_model(model, x_train, y_train, x_test, y_test)  # Evaluate model performance
        print(f"Model: {name}, Metrics: {metrics}")
        plot_confusion_matrix(metrics["Confusion Matrix"], name)  # Visualize results


def train_and_evaluate_nn_model(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Trains and evaluates a deep learning model if available.

    Args:
        x_train: Training features
        y_train: Training labels
        x_val: Validation features
        y_val: Validation labels
        x_test: Testing features
        y_test: Testing labels
    """
    try:
        from models.deep_learning import create_nn_model
        print("Training deep learning model...")

        model = create_nn_model()  # Initialize the neural network model
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, verbose=0)  # Train model

        y_pred_test = model.predict(x_test)  # Predict on test data
        y_hat_test = (y_pred_test >= 0.5).astype(int)  # Convert predictions to binary class

        nn_metrics = evaluate_model(model, x_train, y_train, x_test, y_test)  # Evaluate model
        print("Deep Learning Model Metrics:", nn_metrics)
        plot_confusion_matrix(nn_metrics["Confusion Matrix"], "Neural Network")  # Visualize results
    except Exception as e:
        print(f'Unable to run the deep learning model: {e}')


def run_pipeline():
    """
    Executes the full ML pipeline, including data loading, preprocessing,
    training, and evaluation for both machine learning and deep learning models.
    """
    print("Loading data...")
    data = load_data()  # Load dataset

    print("Preprocessing data...")
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(data)  # Preprocess dataset

    train_and_evaluate_ml_models(x_train, y_train, x_test, y_test)  # Train and evaluate ML models
    train_and_evaluate_nn_model(x_train, y_train, x_val, y_val, x_test, y_test)  # Train and evaluate NN model


if __name__ == "__main__":
    run_pipeline()