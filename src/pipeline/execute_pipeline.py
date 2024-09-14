import pandas as pd
import torch
import timm
from src.preprocessing.import_data import ImportData
from src.preprocessing import preprocess
from src.visualizations.label_distribution import Visualize
from src.training.training_process import TrainModel
from typing import Optional

class PipeLine:
    """
    A class to define the machine learning pipeline, including model training, 
    data preprocessing, and visualization.

    Attributes:
    ----------
    device (torch.device): The device (CPU or GPU) where the model will be trained.
    model_from (Optional[str]): Source from which to load the model ('timm' or '/models').
    model_name (str): The name of the model to load.
    model (Optional[torch.nn.Module]): The model object to be trained.
    train_patience (int): Number of epochs to wait for improvement before stopping training early.
    show_visualizations (bool): Whether to show visualizations of the data.
    df_train (pd.DataFrame): DataFrame containing training data.
    df_val (pd.DataFrame): DataFrame containing validation data.
    df_test (pd.DataFrame): DataFrame containing test data.
    weight_decay (float): Weight decay for regularization.
    learning_rate (float): Learning rate for the optimizer.
    smoothing (float): Label smoothing value.
    decay_t (int): Decay period for learning rate scheduler.
    decay_rate (float): Decay rate for learning rate scheduler.
    warmup_lr_init (float): Initial learning rate for warmup.
    warmup_t (int): Number of warmup epochs.
    noise_range_t (Optional[int]): Range for learning rate noise.
    noise_pct (float): Percentage of noise.
    noise_std (float): Standard deviation of noise.
    noise_seed (int): Random seed for noise generation.
    num_train_epochs (int): Number of epochs for training.
    train_batch_size (int): Batch size for training.
    val_batch_size (int): Batch size for validation.
    test_batch_size (int): Batch size for testing.
    img_size (int): Size of the images (height and width).
    num_workers (int): Number of worker threads for data loading.
    shuffle_train_data (bool): Whether to shuffle training data.
    rebalance_data (bool): Whether to rebalance the training data.
    """
    def __init__(
        self,
        model_from: Optional[str] = None,
        model_name: str = "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k", 
        train_patience: int = 10,
        show_visualizations: bool = False,
        weight_decay: float = 0.0001,
        learning_rate: float = 0.0005,
        smoothing: float = 0.1,
        decay_t: int = 30,
        decay_rate: float = 0.1,
        warmup_lr_init: float = 0.0001,
        warmup_t: int = 3,
        noise_range_t: Optional[int] = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.0,
        noise_seed: int = 42,
        num_train_epochs: int = 10,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        img_size: int = 128,
        num_workers: int = 6,
        shuffle_train_data: bool = True,
        rebalance_data: bool = False
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_from = model_from
        self.model_name = model_name
        self.model = None
        self.train_patience = train_patience
        self.show_visualizations = show_visualizations
        self.rebalance_data = rebalance_data
        self.df_train, self.df_val, self.df_test = ImportData(rebalance_data=self.rebalance_data).import_df() 
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.smoothing = smoothing
        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.shuffle_train_data = shuffle_train_data

    def pipeline_visualizations(self) -> None:
        """
        Generates and shows data visualizations if `show_visualizations` is set to True.

        This method uses the Visualize class to plot different data distributions such as:
        - Original label distribution
        - Modified label distribution
        - Label distribution with minimum cases
        - Binary label distribution
        """
        if self.show_visualizations:
            viz = Visualize(data=pd.concat([self.df_train, self.df_val, self.df_test]))
            viz.plot_label_dist_original()
            viz.plot_label_dist_modified()
            viz.plot_label_dist_min_cases()
            viz.plot_binary_label_dist()

    def train(self) -> None:
        """
        Loads a model from either the 'timm' library or a local directory, 
        prepares data loaders, and trains the model.

        Raises:
        ----------
        ValueError: If `model_from` is not specified or is incorrect.
        """
        # Ensure model source is specified
        if not self.model_from:
            raise ValueError("model_from is not specified. Please provide where to load the model from: {'timm', '/models'}.")
        # Load model from the TIMM library
        if self.model_from == 'timm':
            print(f"Loading model from timm: {self.model_name}.pth")
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=2).to(self.device)
        # Load model from a local directory
        elif self.model_from == '/models':
            model_path = f"./src/models/{self.model_name}.pth"
            print(f"Loading model from /models folder: {model_path}")
            self.model = torch.load(model_path) if self.device.type == "cuda" else torch.load(self.model_path, map_location=torch.device('cpu'))
        else:
             # Raise error if `model_from` is invalid
             raise ValueError("model_from value provided is incorrect. Please provide where to load the model from: {'timm', '/models'}.")

        # Preprocess data and create data loaders
        datasets_loaders = preprocess.ProcessData(
            train_data = self.df_train,
            val_data = self.df_val,
            test_data = self.df_test,
            train_batch_size = self.train_batch_size, 
            val_batch_size = self.val_batch_size, 
            test_batch_size = self.test_batch_size, 
            img_size = self.img_size, 
            num_workers = self.num_workers,
            shuffle_train_data = self.shuffle_train_data
        ).transform_data()

        # Initialize the model trainer
        trainer = TrainModel(
            model = self.model, 
            model_name = self.model_name,
            train_patience = self.train_patience,
            train_loader = datasets_loaders['loaders']['train'], 
            val_loader = datasets_loaders['loaders']['val'],
            device = self.device,
            weight_decay = self.weight_decay,
            learning_rate = self.learning_rate,
            smoothing = self.smoothing,
            decay_t = self.decay_t,
            decay_rate = self.decay_rate,
            warmup_lr_init = self.warmup_lr_init,
            warmup_t = self.warmup_t,
            noise_range_t = self.noise_range_t,
            noise_pct = self.noise_pct,
            noise_std = self.noise_std,
            noise_seed = self.noise_seed,
            num_train_epochs = self.num_train_epochs
        )
        trainer.train()