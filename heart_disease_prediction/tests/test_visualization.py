import unittest
from unittest.mock import patch

# Import the plot_confusion_matrix function from its location.
# Adjust this import path to match where your function is defined.
from heart_disease_prediction.utils.visualization import plot_confusion_matrix


class TestPlotConfusionMatrix(unittest.TestCase):
    # The decorator below patches several matplotlib and seaborn functions that are used
    # within the plot_confusion_matrix function. This is done to intercept calls and verify
    # that they are made correctly, without actually rendering any plots.
    @patch('matplotlib.pyplot.show')      # Prevents the actual plot window from opening
    @patch('matplotlib.pyplot.title')     # Intercepts setting of the plot title
    @patch('matplotlib.pyplot.ylabel')    # Intercepts setting of the y-axis label
    @patch('matplotlib.pyplot.xlabel')    # Intercepts setting of the x-axis label
    @patch('seaborn.heatmap')             # Intercepts the call to seaborn.heatmap to inspect its parameters
    @patch('matplotlib.pyplot.figure')    # Intercepts figure creation to verify the figure size
    def test_plot_confusion_matrix_calls(self, mock_figure, mock_heatmap, mock_xlabel, mock_ylabel, mock_title, mock_show):
        # Arrange: Set up the dummy data and parameters to simulate a typical call.
        dummy_cm = [[10, 2], [3, 5]]  # A dummy confusion matrix for testing purposes.
        model_name = "Test Model"      # A sample model name to appear in the plot title.

        # Act: Call the function under test.
        # This should trigger the patched calls instead of the actual plotting functions.
        plot_confusion_matrix(dummy_cm, model_name)

        # Assert: Check that each of the patched functions was called exactly once with the expected arguments.
        # Verify that the figure was created with a specific size.
        mock_figure.assert_called_once_with(figsize=(5, 4))
        # Verify that the heatmap is drawn with the correct parameters:
        # - The dummy confusion matrix
        # - Annotation enabled (annot=True) with integer formatting (fmt='d')
        # - A blue color map (cmap='Blues')
        # - Custom tick labels for x and y axes ('No' and 'Yes')
        mock_heatmap.assert_called_once_with(
            dummy_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes']
        )
        # Verify that the x-axis is labeled "Predicted"
        mock_xlabel.assert_called_once_with('Predicted')
        # Verify that the y-axis is labeled "Actual"
        mock_ylabel.assert_called_once_with('Actual')
        # Verify that the title is set correctly with the model name included.
        mock_title.assert_called_once_with(f"Confusion Matrix - {model_name}")
        # Verify that the plot is actually displayed by calling plt.show()
        mock_show.assert_called_once()

# This block allows the tests to be run directly from the command line.
if __name__ == '__main__':
    unittest.main()
