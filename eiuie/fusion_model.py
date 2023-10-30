from typing import Optional
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from torch.utils.data import Dataset, DataLoader, random_split

import base_model as bm
import unsharp_masking as um
import retinex as rtx
import homomorphic_filtering as hf
import pixel_dataset as pxds


CHECKPOINT_DIRECTORY = "data/checkpoints"


class FusionNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FusionNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(12, 9),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(9, 6),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    def __init__(
        self, patience=5, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class FusionModel(bm.BaseModel):
    """FusionModel"""

    unsharp_masking: um.UnsharpMasking
    homomorphic_filtering: hf.HomomorphicFiltering
    retinex: rtx.Retinex
    device: torch.device
    net: FusionNet
    optimizer: optim.Optimizer
    criterion: nn.Module
    start_epoch: int = 0

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        unsharp_masking: um.UnsharpMasking = um.UnsharpMasking(),
        homomorphic_filtering: hf.HomomorphicFiltering = hf.HomomorphicFiltering(),
        retinex: rtx.Retinex = rtx.Retinex(),
    ):
        """
        Parameters
        ----------
        checkpoint
            Checkpoint for loading model for inference / resume training.
        """
        self.unsharp_masking = unsharp_masking
        self.homomorphic_filtering = homomorphic_filtering
        self.retinex = retinex

        # Check for GPU availability
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        # Neural Network Model
        self.net = FusionNet()
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.MSELoss()  # assuming regression task
        self.start_epoch = 0

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            latest_checkpoint = self._get_latest_checkpoint()
            if latest_checkpoint:
                self.load_checkpoint(latest_checkpoint)

    def save_checkpoint(self, epoch: int, checkpoint_path: str):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"{CHECKPOINT_DIRECTORY}/{checkpoint_path}",
        )

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(f"{CHECKPOINT_DIRECTORY}/{checkpoint_path}")
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"]

    def _get_latest_checkpoint(self) -> Optional[str]:
        """
        Returns the path to the latest checkpoint from the CHECKPOINT_DIRECTORY.

        Returns
        -------
        Optional[str]
            Path to the latest checkpoint file or None if no checkpoint found.
        """
        if not os.path.exists(CHECKPOINT_DIRECTORY):
            return None
        checkpoint_files = [
            f for f in os.listdir(CHECKPOINT_DIRECTORY) if "checkpoint_epoch_" in f
        ]
        if not checkpoint_files:
            return None

        # Sort based on epoch number
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        return checkpoint_files[-1]  # Return the latest checkpoint

    @property
    def name(self) -> str:
        """
        Name of the model.

        Returns
        -------
        str
            Name of the model.
        """
        return "fusion_model"

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using the model.

        Parameters
        ----------
        image : np.ndarray
            Image to be processed.

        Returns
        -------
        np.ndarray
            Processed image.
        """
        original = bm.BGR2HSI(image)
        um_imagge = self.unsharp_masking.process_image(image)
        um_image = bm.BGR2HSI(um_imagge)
        hf_image = self.homomorphic_filtering.process_image(image)
        hf_image = bm.BGR2HSI(hf_image)
        rtx_image = self.retinex.process_image(image)
        rtx_image = bm.BGR2HSI(rtx_image)

        dimensions = image.shape
        assert dimensions == um_imagge.shape == hf_image.shape == rtx_image.shape

        # Use numpy functions for efficient concatenation
        # Reshape each processed image into (-1, 3), essentially unrolling them
        original = original.reshape(-1, 3)
        um_image = um_image.reshape(-1, 3)
        hf_image = hf_image.reshape(-1, 3)
        rtx_image = rtx_image.reshape(-1, 3)

        # Concatenate them along the horizontal axis (axis=1)
        all_inputs = np.hstack([original, um_image, hf_image, rtx_image])

        # Convert to tensor and move to device
        all_inputs = torch.tensor(all_inputs, dtype=torch.float32).to(self.device)

        # Model inference
        outputs = self.net(all_inputs).cpu().detach().numpy()

        # Reshape outputs back to the original image shape
        fused_image = outputs.reshape(dimensions[0], dimensions[1], 3)
        fused_image = bm.HSI2BGR(fused_image)

        return fused_image

    def train_model(
        self,
        total_epochs=100,
        patience=5,
        train_ratio=0.8,
    ):
        dataset = pxds.PixelDataset()
        # Splitting dataset into training and validation subsets
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=f"{CHECKPOINT_DIRECTORY}/best_model.pt",
        )

        self.net.train()
        for epoch in range(self.start_epoch, total_epochs):
            print()
            print(f"Epoch {epoch+1}/{total_epochs}")
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            # After training, check validation loss
            print("Validating...")
            val_loss = self.validate(val_loader)
            print(f"Validation loss: {val_loss}")

            print("Checking early stopping...")
            early_stopping(val_loss, self.net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Save checkpoint after every epoch
            print("Saving checkpoint...")
            self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pth")

    def validate(self, val_loader):
        self.net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        return average_val_loss
