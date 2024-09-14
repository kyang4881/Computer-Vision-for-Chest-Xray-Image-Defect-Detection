import time
import torch
from src.preprocessing import preprocess
import pandas as pd
from typing import List, Dict, Any

class GeneratePrediction:
    """
    A class to generate predictions from a pre-trained model using a provided test dataset. 
    This class handles loading the model, moving it to the appropriate device (CPU or GPU), 
    and generating predictions.

    Attributes:
    ----------
    device (torch.device): The device (CPU or GPU) where the model will be loaded.
    model (torch.nn.Module): The loaded pre-trained model used for making predictions.
    test_data (pd.DataFrame): The test dataset to be used for generating predictions.
    """
    def __init__(self, model_path: str, test_data: pd.DataFrame) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path).eval() if self.device.type == "cuda" else torch.load(model_path, map_location=torch.device('cpu')).eval()
        self.test_data = test_data

    def predict(self) -> Dict[str, Any]:
        """
        Generates predictions for the test dataset using the pre-trained model.

        Returns:
        ----------
        Dict[str, Any]: A dictionary containing the predictions ('preds') and the time taken to execute ('execution_time').

        Notes:
        ----------
        The method uses torch.no_grad() to disable gradient calculation, which reduces memory consumption for 
        inference. It also records the execution time for processing the entire test dataset.
        """
        # Preprocess the test data using the ProcessData class
        datasets_loaders = preprocess.ProcessData(
            test_data = self.test_data
        ).transform_test_data()

        start_time = time.time()
        preds = []
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Iterate through the test dataset loader
            for images in datasets_loaders['loaders']['test']:
                images = images.to(self.device)
                # Perform forward pass to get model logits
                logits = self.model(images)
                # Get the predicted class indices and convert them to a NumPy array
                preds.extend(torch.max(logits, dim=1)[1].cpu().numpy())  
        # Calculate the time taken for prediction       
        elapsed_time = time.time() - start_time
        # Return the predictions and the elapsed time
        return {
          'preds': preds,
          'execution_time': round(elapsed_time, 2)  # Time in seconds
        }
