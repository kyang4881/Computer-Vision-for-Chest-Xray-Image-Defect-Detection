import torch
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import StepLRScheduler
import torch.optim as optim
from IPython.display import HTML, display
from src.evaluation.metrics import EvaluatePreds
from src.visualizations.utils import plot_train_results
from sklearn.metrics import f1_score
from typing import Optional, List

class TrainModel:
    """
    A class for training a PyTorch model with features like learning rate scheduling, 
    label smoothing, and early stopping.

    Attributes:
    ----------
    model (torch.nn.Module): The model to be trained.
    model_name (str): Name of the model, used for saving and loading.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    device (torch.device): Device on which to run the training (CPU or GPU).
    train_patience (int): Number of epochs with no improvement before stopping early.
    weight_decay (float): Weight decay for the optimizer.
    learning_rate (float): Learning rate for the optimizer.
    smoothing (float): Label smoothing parameter.
    decay_t (int): Number of epochs before learning rate decay.
    decay_rate (float): Factor by which learning rate is decayed.
    warmup_lr_init (float): Initial learning rate during warmup phase.
    warmup_t (int): Number of epochs for warmup phase.
    noise_range_t (Optional[float]): Range of noise added to the learning rate.
    noise_pct (float): Percentage of noise applied to the learning rate.
    noise_std (float): Standard deviation of noise applied to the learning rate.
    noise_seed (int): Seed for random number generation of noise.
    folder_path (str): Directory where model checkpoints are saved.
    num_train_epochs (int): Number of epochs for training.
    """
    def __init__(self, 
                 model, 
                 model_name,
                 train_loader, 
                 val_loader,
                 device,
                 train_patience,
                 weight_decay = 0.0001,
                 learning_rate = 0.0005,
                 smoothing = 0.1,
                 decay_t = 30,
                 decay_rate = 0.1,
                 warmup_lr_init = 0.0001,
                 warmup_t = 3,
                 noise_range_t = None,
                 noise_pct = 0.67,
                 noise_std = 1.,
                 noise_seed = 42,
                 folder_path = "./src/data/models/",
                 num_train_epochs = 10
        ):
        self.model = model
        self.model_name = model_name
        self.train_patience = train_patience
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
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
        self.folder_path = folder_path
        self.num_train_epochs = num_train_epochs

    def progress(self, value, max=100):
        """
        Creates an HTML progress bar to visualize training progress.

        Args:
        ----------
        value (int): Current progress value.
        max (int, optional): Maximum value for the progress bar (default is 100).

        Returns:
        ----------
        HTML: An HTML representation of the progress bar.
        """
        return HTML("""
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(value=value, max=max))
    
    def train(self) -> None:
        """
        Trains the model using the specified training and validation data loaders. 

        Performs the following tasks:
        - Initializes the optimizer and loss function.
        - Sets up the learning rate scheduler.
        - Iterates over the specified number of epochs to train the model.
        - Saves the best model based on validation performance.
        - Implements early stopping based on validation performance.
        """
        # Initialize optimizer and loss function
        optimizer = optim.AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        loss_fn = LabelSmoothingCrossEntropy(smoothing=self.smoothing).cuda()
        # Set up learning rate scheduler
        lr_scheduler = StepLRScheduler(
            optimizer, 
            decay_t = self.decay_t, 
            decay_rate = self.decay_rate, 
            warmup_lr_init = self.warmup_lr_init, 
            warmup_t = self.warmup_t, 
            noise_range_t = self.noise_range_t, 
            noise_pct = self.noise_pct, 
            noise_std = self.noise_std, 
            noise_seed = self.noise_seed
        )
        # Initialize progress bar for training
        progress_bar_train = display(self.progress(0, self.num_train_epochs), display_id=True)
        # Lists to store training and validation metrics
        train_losses = []
        train_score = []
        val_score = []

        patience_cnt = 0

        for epoch in range(self.num_train_epochs):
            batch_losses = []
            preds = []
            targets = []
            # Load the best score from file if it exists
            try:
                with open(f"./src/models/{self.model_name}_best_score.txt", 'r') as file:
                    content = file.read()
                    best_score = float(content)
                    #print(f"Loaded the current best score from file: {best_score}")
                file.close()
            except:
                print(f"Score file not found: ./src/models/{self.model_name}_best_score.txt")
                best_score = 0.0
             # Training loop
            for images, target in self.train_loader:
                images, target = images.to(self.device), target.to(self.device)
                logits = self.model(images)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_losses.append(loss.item()) # use ".item()" to get the score from the tensor
                # Collect predictions and targets
                with torch.no_grad():
                    preds.extend(torch.max(logits, dim=1)[1].cpu().numpy())
                    targets.extend(target.cpu().numpy())
            # Store training metrics
            train_losses.append(np.mean(batch_losses))
            train_score.append(f1_score(targets, preds, average='macro'))
            # Increment step of the learning rate scheduler
            lr_scheduler.step(epoch + 1)
            # Evaluate on validation set and store metrics
            val_score.append(EvaluatePreds(model = self.model, loader = self.val_loader, device = self.device).evaluate()['f1_score'])

            if len(val_score) > 1:
                plot_train_results(train_losses, train_score, val_score)  

            # Save the best model based on validation performance
            if val_score[-1] > best_score:
                patience_cnt = 0
                print(f"A better model is now available. The current best score is: {val_score[-1]}, The previous best score is: {best_score}")
                torch.save(self.model, f"./src/models/{self.model_name}.pth")

                # Save best score
                file_path = f"./src/models/{self.model_name}_best_score.txt"
                with open(file_path, 'w') as file:
                    file.write(str(val_score[-1]))
                    print(f"Saved the current best score to the file.")
                file.close()
            else:
                patience_cnt += 1

            self.model.train()

            progress_bar_train.update(self.progress(epoch+1, self.num_train_epochs))
            print(
                f'\r[Iteration {epoch+1}] loss={round(train_losses[-1], 2)} | '
                f'train score = {round(train_score[-1]*100, 2)}% | '
                f'validation score = {round(val_score[-1]*100, 2)}% | '
                f'best validation score = {round(best_score*100, 2)}% | '
                , flush=True)
            # Break the training loop if the max number of epochs without improvement in validation score is reached
            if patience_cnt >= self.train_patience:
                print('Early stopping triggered')
                break