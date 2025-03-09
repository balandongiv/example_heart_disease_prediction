from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression model from scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Importing Random Forest Classifier from scikit-learn
from sklearn.tree import DecisionTreeClassifier  # Importing Decision Tree Classifier from scikit-learn

def get_models():
    """
    This function returns a dictionary of machine learning models.
    Each key in the dictionary represents the model name (as a string),
    and the corresponding value is an instance of the model from scikit-learn.

    Returns:
        dict: A dictionary containing model names as keys and their respective instantiated objects as values.
    """
    return {
        "Logistic Regression": LogisticRegression(),  # Instantiating a Logistic Regression model
        "Decision Tree": DecisionTreeClassifier(),  # Instantiating a Decision Tree model
        "Random Forest": RandomForestClassifier()  # Instantiating a Random Forest model
    }