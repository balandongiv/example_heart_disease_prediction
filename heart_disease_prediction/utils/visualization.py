import matplotlib.pyplot as plt  # Importing Matplotlib for visualization
import seaborn as sns  # Importing Seaborn for enhanced visualizations

def plot_confusion_matrix(cm, model_name):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    Parameters:
    cm (array-like): The confusion matrix to be visualized.
    model_name (str): Name of the model, used in the title.

    Returns:
    None: Displays the heatmap.
    """

    plt.figure(figsize=(5, 4))  # Set the figure size to 5x4 inches

    # Creating a heatmap for the confusion matrix
    # annot=True adds numerical values to the heatmap cells
    # fmt='d' ensures that the numbers are displayed as integers
    # cmap='Blues' sets the color scheme to shades of blue
    # xticklabels and yticklabels label the axes with 'No' and 'Yes'
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])

    plt.xlabel('Predicted')  # Label the x-axis as 'Predicted'
    plt.ylabel('Actual')  # Label the y-axis as 'Actual'
    plt.title(f"Confusion Matrix - {model_name}")  # Set the title with the model name
    plt.show()  # Display the heatmap
