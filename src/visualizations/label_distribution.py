import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from collections import Counter
import pandas as pd
from typing import Optional

class Visualize:
    """
    A class for visualizing label distributions in a dataset.

    Attributes:
    ----------
    data (pd.DataFrame): The dataset to visualize.
    verbose (bool): Whether to print additional information during visualization.
    data_modified (Optional[pd.DataFrame]): The dataset after modifications for visualization purposes.
    """
    def __init__(self, data: pd.DataFrame, verbose: bool = True) -> None:
        self.data = data
        self.verbose = verbose
        self.data_modified = None

    def plot_label_dist_original(self) -> None:
        """
        Plots the distribution of the original labels in the dataset.
        """
        if self.verbose: print("\n\nDistribution of the original labels")
        # Count occurrences of each original label
        label_counts = self.data['original_label'].value_counts()[:50]
        fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
        ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
        ax1.set_xticks(np.arange(len(label_counts))+0.5)
        _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
        plt.show()

    def plot_label_dist_modified(self) -> None:
        """
        Plots the distribution of the modified labels after sampling based on original label frequencies.
        """
        if self.verbose: print("\n\nDistribution of the mapped labels")
        # Calculates the sample weights based on the number of categories in each label
        # For each label, len(x.split('|')) counts how many categories it has. 
        # If a label has one or more categories, it adds a small constant (4e-2, or 0.04) to ensure that no label gets a weight of zero.
        sample_weights = self.data['original_label'].map(lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2 
        # Result is normalized by dividing each weight by the sum of all weights, ensuring that the total weight sum to 1.
        sample_weights /= sample_weights.sum() 
        sample_size = len(self.data)
        # Sample the dataset with weights
        # Reason: the original labels have multiple diseases combined using "|", by resampling with the weights
        # the labels that are more common (considering those with multiple diseases) will be choosen more often
        self.data_modified = self.data.sample(sample_size, weights=sample_weights)
        # After sampling, count how often each label occurs in the modified dataset and select the top 15 most frequent labels.
        label_counts = self.data_modified['original_label'].value_counts()[:15]
        fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
        ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
        ax1.set_xticks(np.arange(len(label_counts))+0.5)
        _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
        plt.show()

    # only consider labels that have atleast 10 cases
    def plot_label_dist_min_cases(self, min_cases: int = 10) -> None:
        """
        Plots the distribution of labels with at least a minimum number of cases.

        Args:
        ----------
        min_cases (int, optional): Minimum number of cases for a label to be considered.
        """
        if self.verbose: print(f"\n\nDistribution of the mapped labels with atleast {min_cases} cases")
        # Extract all unique labels from the modified dataset
        all_labels = np.unique(list(chain(*self.data_modified['original_label'].map(lambda x: x.split('|')).tolist())))
        all_labels = [x for x in all_labels if len(x)>0]
        if self.verbose: print('All Labels ({}): {}'.format(len(all_labels), all_labels))
        # Create binary columns for each label
        for c_label in all_labels:
            if len(c_label)>1: 
                self.data_modified[c_label] = self.data_modified['original_label'].map(lambda x: 1.0 if c_label in x else 0)
        if self.verbose: 
            print("\n\nMapped Multi-Label Occurences:")
            display(self.data_modified.sample(3))
        # Filter labels with at least `min_cases`
        all_labels = [c_label for c_label in all_labels if self.data_modified[c_label].sum() > min_cases]
        if self.verbose: print('Clean Labels ({})'.format(len(all_labels)),[(c_label, int(self.data_modified[c_label].sum())) for c_label in all_labels])
        # Calculate the percentage of cases for each label (Note that sum of the percentages > 100% due to the overlapping in labels)
        label_counts = 100*np.mean(self.data_modified[all_labels].values, 0)
        # Sort the labels and their corresponding values in descending order
        sorted_indices = np.argsort(label_counts)[::-1]  
        sorted_labels = [all_labels[i] for i in sorted_indices]
        sorted_label_counts = label_counts[sorted_indices]
        # Plotting the sorted labels and their frequencies
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.bar(np.arange(len(sorted_label_counts)) + 0.5, sorted_label_counts)
        ax1.set_xticks(np.arange(len(sorted_label_counts)) + 0.5)
        ax1.set_xticklabels(sorted_labels, rotation=90)
        ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
        _ = ax1.set_ylabel('Frequency (%)')
        plt.show()

    def plot_binary_label_dist(self) -> None:
        """
        Plots the distribution of binary labels (e.g., presence or absence of a disease).
        """
        if self.verbose: print("\n\nDistribution of the mapped binary labels")
        # Count occurrences of binary labels
        data = Counter(self.data_modified['label'])
        label_map = {'0': 'No Finding', '1': 'Disease'}
        mapped_labels = [label_map[label] for label in data.keys()]
        values = list(data.values())
        plt.figure(figsize=(6, 4))
        plt.bar(mapped_labels, values, color=['lightcoral', 'skyblue'])
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.title('Distribution of Labels')
        plt.show()