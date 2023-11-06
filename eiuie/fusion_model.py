from typing import Optional, Callable, Any, Tuple, List, Dict
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from torch.utils.data import DataLoader, random_split

import base_model as bm
import unsharp_masking as um
import retinex as rtx
import homomorphic_filtering as hf
import pixel_dataset as pxds


CHECKPOINT_DIRECTORY = "data/checkpoints"


class ChannelNet(nn.Module):
    """Single layer perceptron for individual channels."""

    def __init__(self, input_size: int):
        super(ChannelNet, self).__init__()
        self.fc = nn.Linear(input_size, 1, bias=False)

    def forward(self, x):
        return self.fc(x)

    def get_params(self) -> Tuple[List[float], float]:
        # Weights, bias
        weights, bias = self.fc.weight.data.numpy().flatten(), self.fc.bias.data.numpy()
        # as python objects
        weights, bias = weights.tolist(), bias.tolist()
        return weights, bias


class FusionNet(nn.Module):
    """Unifying model for all channels."""

    use_original: bool

    def __init__(self, use_original: bool):
        super(FusionNet, self).__init__()
        self.use_original = use_original
        self.h_net = ChannelNet(4 if use_original else 3)
        self.s_net = ChannelNet(4 if use_original else 3)
        self.i_net = ChannelNet(4 if use_original else 3)

    def forward(self, x):
        # Flatten the middle dimensions
        x = x.view(-1, 12)  # This will reshape the input to (batch_size, 12)

        # Splitting the input for the three channels
        h_channel = x[
            :, 0 if self.use_original else 3 :: 3
        ]  # Every third value starting from index 0
        s_channel = x[
            :, 1 if self.use_original else 4 :: 3
        ]  # Every third value starting from index 1
        i_channel = x[
            :, 2 if self.use_original else 5 :: 3
        ]  # Every third value starting from index 2

        # Getting the outputs
        h_out = self.h_net(h_channel)
        s_out = self.s_net(s_channel)
        i_out = self.i_net(i_channel)

        # Concatenate the outputs to get the final output
        return torch.cat((h_out, s_out, i_out), dim=1)

    def get_params(self) -> Dict[str, Tuple[List[float], float]]:
        return {
            "h": self.h_net.get_params(),
            "s": self.s_net.get_params(),
            "i": self.i_net.get_params(),
        }


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    patience: int
    verbose: bool
    counter: int
    best_score: Optional[float]
    early_stop: bool
    val_loss_min: float
    delta: float
    trace_func: Callable[[Any], None]

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0.0,
        trace_func: Callable[[Any], None] = print,
    ):
        """
        Parameters
        ----------
        patience
            How long to wait after last time validation loss improved.
        verbose
            If True, prints a message for each validation loss improvement.
        delta
            Minimum change in the monitored quantity to qualify as an improvement.
        trace_func
            Function to trace the message.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss: float):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


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
        self.net = FusionNet(use_original=False).to(self.device)
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
        # Ensure checkpoint directory exists
        if not os.path.exists(CHECKPOINT_DIRECTORY):
            os.makedirs(CHECKPOINT_DIRECTORY)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"{CHECKPOINT_DIRECTORY}/{checkpoint_path}",
        )

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(
            f"{CHECKPOINT_DIRECTORY}/{checkpoint_path}", map_location=self.device
        )
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
            f
            for f in os.listdir(CHECKPOINT_DIRECTORY)
            if "checkpoint_epoch_" in f or "best_model" in f
        ]
        if not checkpoint_files:
            return None

        if "best_model.pt" in checkpoint_files:
            return "best_model.pt"

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

        # All values between 0 and 1
        outputs = np.clip(outputs, 0, 1)

        # Reshape outputs back to the original image shape
        fused_image = outputs.reshape(dimensions[0], dimensions[1], 3)
        fused_image = bm.HSI2BGR(fused_image)

        return fused_image

    def train_model(
        self,
        total_epochs: int = 50,
        patience: int = 5,
        data_to_use: float = 0.005,
        train_ratio: float = 0.8,
        batch_size: int = 1024,
        pre_shuffle: bool = False,
        shuffle: bool = True,
    ):
        print("Loading dataset...")
        dataset = pxds.PixelDataset(
            use_fraction=data_to_use,
            use_exposures="both",
            batch_size=batch_size,
            pre_shuffle=pre_shuffle,
            shuffle_batch=shuffle,
        )
        # Splitting dataset into training and validation subsets
        print("Splitting dataset into training and validation subsets...")
        data_len = len(dataset)
        print("Data points to use:", data_len)
        train_size = int(train_ratio * data_len)
        print("Training data points:", train_size)
        val_size = data_len - train_size
        print("Validation data points:", val_size)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=lambda x: x[0],
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=0,
            pin_memory=True,
        )

        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
        )

        best_val_loss = float("inf")

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best model...")
                self.save_checkpoint(epoch, "best_model.pt")

            print("Checking early stopping...")
            early_stopping(val_loss)

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

    def get_parameters(self):
        return self.net.parameters()

    def pretty_print_parameters(self):
        print("Model parameters:")
        for channel, (weights, bias) in self.net.get_params().items():
            print(f"Channel {channel}:")
            print("Weights:")
            print(weights)
            print("Bias:")
            print(bias)
            print()
