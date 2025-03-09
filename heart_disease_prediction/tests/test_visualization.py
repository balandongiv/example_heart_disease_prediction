import os
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import plot_confusion_matrix

def test_plot_confusion_matrix_saves_file():
    """
    Test that plot_confusion_matrix correctly saves an image when a save path is provided.
    """
    cm = np.array([[50, 10], [5, 35]])  # Example confusion matrix
    save_path = "test_confusion_matrix.png"

    # Ensure the file does not exist before the test
    if os.path.exists(save_path):
        os.remove(save_path)

    # Call the function with a save path
    plot_confusion_matrix(cm, "Test Model", save_path=save_path)

    # Check if the file was created
    assert os.path.exists(save_path), "Confusion matrix image was not saved."

    # Cleanup: Remove the test file after checking
    os.remove(save_path)

def test_plot_confusion_matrix_displays():
    """
    Test that plot_confusion_matrix runs without errors when no save path is provided.
    """
    cm = np.array([[30, 20], [10, 40]])  # Example confusion matrix

    try:
        plot_confusion_matrix(cm, "Test Model")  # Should display the plot
        plt.close()  # Close the plot to prevent display issues
    except Exception as e:
        assert False, f"plot_confusion_matrix raised an exception: {e}"
