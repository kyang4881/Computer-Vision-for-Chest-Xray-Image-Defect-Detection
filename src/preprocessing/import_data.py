import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class ImportData:
    """
    A class to handle importing and processing data, including reading image file paths, 
    labels, and splitting the data into training, validation, and test sets.

    Attributes:
    ----------
    label_df_path (str): Path to the CSV file containing image labels and other metadata.
    txt_files (List[str]): List of text files specifying the images to be used for training, validation, and testing.
    n_image_folders (int): Number of image folders to scan for images.
    """
    def __init__(
        self, 
        label_df_path: str = "./src/data/Data_Entry_2017_v2020.csv", 
        txt_files: List[str] = ["test_list", "train_val_list"],
        n_image_folders: int = 2,
        rebalance_data: bool = False
    ) -> None:
        self.label_df_path = label_df_path
        self.txt_files = txt_files
        self.n_image_folders = n_image_folders
        self.rebalance_data = rebalance_data
        
    def data_entry_df(self, data: pd.DataFrame, type_of_data: str = None) -> pd.DataFrame:
        """
        Generates a DataFrame containing the updated image classes, 
        file paths, file names, and original image classes

        Args:
        ----------
        data (pd.DataFrame): DataFrame containing image indices and their corresponding labels.
        type_of_data (str): String indicating whether Training/Validation or Testing data.

        Returns:
        ----------
        pd.DataFrame: A DataFrame with columns for image labels, file paths, filenames, and original labels.
        """
        file_dict = {}
        # Loop through each image folder and map image filenames to their full paths
        for i in range(1, self.n_image_folders + 1):
            # List all image filenames in the folder and sort them
            file_names = np.sort(os.listdir(f'./src/data/images_00{str(i)}/images/')).tolist()
             # Create full file paths for the images
            file_paths = np.sort(list(map(lambda x: f"images_00{str(i)}" + "/images/" + x, file_names)))
            # Update the dictionary with filenames as keys and paths as values
            file_dict.update(dict(zip(file_names, file_paths)))
        # Map labels to '0' for 'No Finding' and '1' for other labels
        mapped_labels = list(map(lambda x: '0' if x=='No Finding' else '1', data['Finding Labels']))
        # Create and return a DataFrame with image labels, file paths, filenames, and original labels
        not_balanced_df = pd.DataFrame({
            'label': mapped_labels, 
            'file_path': data['Image Index'].map(lambda x: file_dict[x]), 
            'filename': data['Image Index'], 
            'original_label': data['Finding Labels']
        })    
        # Rebalancing dataset
        if self.rebalance_data and type_of_data == "Train":
            # Get the count of each class
            class_counts = not_balanced_df['label'].value_counts()
            # Determine the minimum class count
            min_class_count = class_counts.min()
            # Sample from each class to match the size of the smallest class
            balanced_df = pd.concat([
                not_balanced_df[not_balanced_df['label'] == '0'].sample(n=min_class_count, random_state=42),
                not_balanced_df[not_balanced_df['label'] == '1'].sample(n=min_class_count, random_state=42)
            ]).reset_index(drop=True)

            return balanced_df
        
        return not_balanced_df

    def import_df(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Imports data from the specified text files, and processes them to create the
        training, validation, and test DataFrames.

        Returns:
        ----------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the DataFrames for training, validation, and test sets.
        """
        data = []
        for file in self.txt_files:
            with open("./src/data/" + file + ".txt", 'r') as f:
                data.append(f.read().splitlines())
        # Read the label DataFrame from the CSV file
        label_df = pd.read_csv(self.label_df_path)
        # Filter the DataFrame for training/validation and test images
        train_val_df = label_df[label_df['Image Index'].isin(data[1])].reset_index(drop=True)
        test_df = label_df[label_df['Image Index'].isin(data[0])].reset_index(drop=True)
        # Generate DataFrames mapping file paths and labels for training/validation and test datasets
        df_train_set = self.data_entry_df(data = train_val_df, type_of_data = "Train")
        df_test = self.data_entry_df(data = test_df, type_of_data = "Test")
        # Split the training set into training and validation sets
        df_train, df_val = train_test_split(df_train_set, test_size=0.2, random_state=123)
        df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)

        return df_train, df_val, df_test
