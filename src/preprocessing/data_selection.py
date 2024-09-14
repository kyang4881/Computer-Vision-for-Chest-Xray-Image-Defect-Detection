import os
from PIL import Image
import torch
from typing import Any, Optional, Tuple

class GetData(torch.utils.data.Dataset):
    """
    A class to handle image data loading for training, validation, and testing in a pipeline. 

    Attributes:
    ----------
    dataframe (DataFrame): DataFrame containing image file paths and labels.
    data_dir (str): Directory location where the image data is stored.
    transform (Optional[callable]): A function/transform that takes in an image and returns a transformed version.
    type_of_data (str): Specifies the type of data ("Train", "Validation", "Test") to determine whether labels are required.
    """
    def __init__(self, dataframe: Any, data_dir: str, transform: Optional[Any] = None, type_of_data: str = "Train") -> None:
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.type_of_data = type_of_data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        ----------
        int: The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        This function fetches an image and its corresponding label (if applicable) based on the provided index:

        -Retrieves the file path from the dataframe using the index.
        -Constructs the full file path if a data_dir is specified.
        -Opens the image using PIL.Image.open(), converting it to RGB mode.
        -Applies transformations (e.g., resizing, augmentations) if the transform argument was passed.
        -If type_of_data is "Train", it fetches the label for that image, converts it to an integer, and 
            returns both the image and label.
        -If type_of_data is not "Train", it only returns the image, making it suitable for use during 
            validation or testing when labels may not be required.
        """

        """
        Fetches an image and its corresponding label (if not test data) based on the provided index.

        Args:
        ----------
        idx (int): Index of the sample to fetch.

        Returns:
        ----------
        Tuple[Any, Optional[int]]: A tuple containing the transformed image and its label (if `type_of_data` is "Train"), 
        or just the image if `type_of_data` is not "Train".
        """
        # Retrieve the file path from the dataframe using the index
        file_name = self.dataframe.iloc[idx].file_path
        # Construct the full file path if a data directory is specified
        if self.data_dir is not None:
            file_name = os.path.join(self.data_dir, file_name)
        # Open the image using PIL, converting it to RGB mode
        image = Image.open(file_name).convert('RGB')
        # Apply transformations if specified
        if self.transform: image = self.transform(image)
        # Fetch and return the label if type_of_data is "Train"
        if self.type_of_data == "Train": 
            label = int(self.dataframe.iloc[idx].label)
            return image, label
        else:
            # Return only the image for test data
            return image

    