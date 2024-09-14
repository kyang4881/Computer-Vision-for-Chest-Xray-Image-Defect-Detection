from torchvision import transforms
from torch.utils.data import DataLoader
from src.preprocessing.data_selection import GetData
from typing import Dict, Any, Optional

class ProcessData:
    """
    A class to handle preprocessing and data loading for training, validation, 
    and testing datasets in a deep learning pipeline.

    Attributes:
    ----------
    train_batch_size (int): Batch size for training data.
    val_batch_size (int): Batch size for validation data.
    test_batch_size (int): Batch size for test data.
    img_size (int): Size to which images will be resized.
    num_workers (int): Number of worker processes for loading data.
    shuffle_train_data (bool): Whether to shuffle the training data.
    train_data (Optional[DataFrame]): DataFrame containing training data information (e.g., paths and labels).
    val_data (Optional[DataFrame]): DataFrame containing validation data information.
    test_data (Optional[DataFrame]): DataFrame containing test data information.
    data_file_location (str): Directory location where the image data is stored.
    """
    def __init__(self, 
                 train_batch_size: int = 64, 
                 val_batch_size: int = 64, 
                 test_batch_size: int = 64, 
                 img_size: int = 128, 
                 num_workers: int = 6,
                 shuffle_train_data: bool = True,
                 train_data: Optional[Any] = None, 
                 val_data: Optional[Any] = None, 
                 test_data: Optional[Any] = None, 
                 data_file_location: str = "./src/data/"
        ) -> None:
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.img_size = img_size
        self.data_file_location = data_file_location
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.shuffle_train_data = shuffle_train_data

    def transform_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Preprocesses and transforms training and validation data, and creates DataLoader objects.

        Returns:
        -------
        dict: A dictionary containing datasets and data loaders for training and validation data.

        Raises:
        ------
        ValueError: If either `train_data` or `val_data` is None.
        """
        # Ensure both training and validation data are provided
        if self.train_data is None or self.val_data is None:
            raise ValueError("train_data and val_data cannot be None")

        # Define transformations for training data
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Define transformations for validation data
        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            #transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Create datasets for training and validation using GetData
        train_dataset = GetData(
            dataframe=self.train_data, 
            data_dir=self.data_file_location, 
            transform=train_transform
        )
        val_dataset = GetData(
            dataframe=self.val_data, 
            data_dir=self.data_file_location, 
            transform=val_transform
        )
        # Create data loaders for training and validation
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=self.shuffle_train_data, 
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.val_batch_size, 
            num_workers=self.num_workers
        )
        # Return datasets and data loaders
        return {"datasets": {'train': train_dataset, 'val': val_dataset}, 
                "loaders": {'train': train_loader, 'val': val_loader}}
    
    def transform_test_data(self):
        """
        Preprocesses and transforms the test data.

        Returns:
        -------
        dict: A dictionary containing the dataset and data loader for test data.

        Raises:
        ------
        ValueError: If `test_data` is None.
        """
        # Ensure test data is provided
        if self.test_data is None:
            raise ValueError("test_data cannot be None")

        # Define transformations for test data
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Create dataset for test data using GetData
        test_dataset = GetData(
            dataframe=self.test_data, 
            data_dir=self.data_file_location, 
            transform=test_transform,
            type_of_data="Test"
        )
        # Create data loader for test data
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.test_batch_size, 
            num_workers=self.num_workers
        )
        # Return dataset and data loader
        return {"datasets": {'test': test_dataset}, 
                "loaders": {'test': test_loader}}
    

# 1. Purpose of DataLoader:
# The DataLoader in PyTorch helps in:

# Batching: Automatically grouping data samples into batches.
# Shuffling: Randomizing the order of data to reduce overfitting.
# Parallel Loading: Loading data in parallel with multiple workers, speeding up the data loading process.