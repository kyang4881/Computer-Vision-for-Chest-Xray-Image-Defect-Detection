import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from typing import List, Tuple, Union

def show_image(data: Union[List[Tuple[torch.Tensor, int]], List[Tuple[np.ndarray, int]]], index: int) -> None:
    """
    Displays an image from the dataset with its corresponding label.

    Args:
    ----------
    data (Union[List[Tuple[torch.Tensor, int]], List[Tuple[np.ndarray, int]]]): 
        The dataset containing image-label pairs.
    index (int): 
        The index of the image in the dataset to display.
    """
    image, label = data[index]
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0)  
        image = image.numpy()
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

def plot_train_results(losses: List[float], score_train: List[float], score_val: List[float]) -> None:
    """
    Plots the training progress including losses and scores.

    Args:
    ----------
    losses (List[float]): List of training losses for each epoch.
    score_train (List[float]): List of training F1 scores for each epoch.
    score_val (List[float]): List of validation F1 scores for each epoch.
    """
    # Clear the previous output in Jupyter
    clear_output(wait=True)  
    # Generate a list of epochs (1 to N)
    epochs = list(range(1, len(losses) + 1))  
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Train and Validation Scores on the left y-axis
    ax1.plot(epochs, score_train, label='Train Score', color='green')
    ax1.plot(epochs, score_val, label='Validation Score', color='blue')
    ax1.set_ylabel('F1 Score')
    ax1.set_xlabel('Epoch')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1)  # Set F1 Score range between 0 and 1
    ax1.set_yticks([i * 0.05 for i in range(21)])  # Fine-grained y-ticks for F1
    
    # Annotate the best validation score
    max_val_score = max(score_val)
    max_val_epoch = score_val.index(max_val_score) + 1  # Offset for 1-based epoch
    ax1.annotate(f'Current Best Validation Score: {max_val_score:.2%}', 
                 xy=(max_val_epoch, max_val_score), 
                 xytext=(max_val_epoch, max_val_score + 0.05), 
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, color='blue')
    
    # Plot Train Loss on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, losses, label='Train Loss', color='red', linestyle='--')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')  # Logarithmic scale for loss
    ax2.tick_params(axis='y')
    ax2.set_ylim(min(losses), max(losses))  # Set loss range
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()