import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Import the function to test.
# Adjust the import path according to your project's structure.
from heart_disease_prediction.models.ml_models import get_models

def test_get_models_returns_dict():
    """
    Test that get_models() returns a dictionary.

    This test verifies that the function returns a data type of dictionary,
    which is the expected output format containing the models.
    """
    models = get_models()
    assert isinstance(models, dict), "Expected a dictionary of models."

def test_get_models_expected_keys():
    """
    Test that the dictionary returned by get_models() contains the expected model names.

    The expected keys are:
      - "Logistic Regression"
      - "Decision Tree"
      - "Random Forest"

    This test compares the set of keys in the returned dictionary with the expected keys.
    """
    models = get_models()
    expected_keys = {"Logistic Regression", "Decision Tree", "Random Forest"}
    # Compare the expected keys with the keys from the dictionary.
    assert expected_keys == set(models.keys()), (
        f"Expected keys {expected_keys}, but got {set(models.keys())}"
    )

def test_get_models_correct_types():
    """
    Test that each key in the dictionary corresponds to the correct scikit-learn model instance.

    This test ensures that:
      - The "Logistic Regression" key holds an instance of LogisticRegression.
      - The "Decision Tree" key holds an instance of DecisionTreeClassifier.
      - The "Random Forest" key holds an instance of RandomForestClassifier.
    """
    models = get_models()

    # Verify the model type for "Logistic Regression"
    assert isinstance(models["Logistic Regression"], LogisticRegression), (
        "The 'Logistic Regression' key should be an instance of LogisticRegression."
    )
    # Verify the model type for "Decision Tree"
    assert isinstance(models["Decision Tree"], DecisionTreeClassifier), (
        "The 'Decision Tree' key should be an instance of DecisionTreeClassifier."
    )
    # Verify the model type for "Random Forest"
    assert isinstance(models["Random Forest"], RandomForestClassifier), (
        "The 'Random Forest' key should be an instance of RandomForestClassifier."
    )
