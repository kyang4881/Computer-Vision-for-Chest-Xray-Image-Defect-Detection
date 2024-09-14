import time
import torch
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

class EvaluatePreds:
    """
    A class to evaluate the performance of a trained model on a given dataset.

    Attributes:
    ----------
    model (torch.nn.Module): The model to be evaluated.
    loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
    device (torch.device): Device on which to run the evaluation (CPU or GPU).
    """
    def __init__(self, model, loader, device):
        self.model = model.eval()
        self.loader = loader
        self.device = device

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the model on the provided dataset and calculates metrics such as F1 score and accuracy.

        Returns:
        ----------
        Dict[str, Any]: A dictionary containing evaluation metrics and predictions:
            - 'f1_score' (float): Macro F1 score of the predictions.
            - 'accuracy_score' (float): Accuracy score of the predictions.
            - 'preds' (List[int]): List of predicted labels.
            - 'targets' (List[int]): List of true labels.
            - 'execution_time' (float): Time taken to perform the evaluation in seconds.
        """
        start_time = time.time()
        preds = []
        targets = []
        # Disable gradient calculation for evaluation
        with torch.no_grad():
            for images, target in self.loader:
                images, target = images.to(self.device), target.to(self.device)
                # Get model predictions (logits) for the batch
                logits = self.model(images)
                # Convert logits to predicted labels and store them
                preds.extend(torch.max(logits, dim=1)[1].cpu().numpy())  
                # Store the true labels
                targets.extend(target.cpu().numpy())  
        # Calculate the time taken for evaluation
        elapsed_time = time.time() - start_time
        # Return evaluation metrics, predictions, true labels, and execution time
        return {
          'f1_score': round(f1_score(targets, preds, average='macro'), 2),
          'accuracy_score': round(accuracy_score(targets, preds), 2),
          'preds': preds,
          'targets': targets,
          'execution_time': round(elapsed_time, 2)  # Time in seconds
        }
     