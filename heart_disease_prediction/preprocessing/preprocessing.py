from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import configuration constants for test size, validation size, and random state
from heart_disease_prediction.config import TEST_SIZE, VAL_SIZE, RANDOM_STATE

def preprocess_data(data):
    """
    Preprocesses the given dataset by performing the following steps:
    1. Splits features (X) and target variable (y).
    2. Splits the dataset into training, validation, and test sets.
    3. Standardizes the feature variables using StandardScaler.

    Parameters:
    data (DataFrame): Input dataset containing features and target variable.

    Returns:
    Tuple: Preprocessed training, validation, and test sets (X and y components).
    """

    # Separate features (X) and target variable (y)
    x = data.drop("Heart_Risk", axis=1)  # Drop the target column to retain feature variables
    y = data["Heart_Risk"]  # Extract the target variable

    # Split the dataset into training and remaining (validation + test) sets
    x_train, x_, y_train, y_ = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Further split the remaining data into validation and test sets
    x_val, x_test, y_val, y_test = train_test_split(
        x_, y_, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Initialize the StandardScaler for feature standardization
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    x_train = scaler.fit_transform(x_train)

    # Use the fitted scaler to transform validation and test sets (without refitting)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Return the preprocessed datasets
    return x_train, x_val, x_test, y_train, y_val, y_test
