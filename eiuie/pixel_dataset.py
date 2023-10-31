from typing import Tuple, Literal
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import base_model as bm

FILE_LOW = "data/pixel_dataset/low.ds"
FILE_HIGH = "data/pixel_dataset/high.ds"


class PixelDataset(Dataset):
    def __init__(
        self,
        chunk_size: int = 10000,
        use_fraction: float = 0.5,
        use_exposures: Literal["high", "low", "both"] = "both",
        pre_shuffle: int = False,
        shuffle_batch: int = True,
        batch_size: int = 1,
    ):
        # Ensure use_fraction is within valid bounds
        use_fraction = max(0.0, min(1.0, use_fraction))

        # Use numpy's memory mapping
        f = (0.25 + random.random() * 0.5) * use_fraction
        fractions = [f, use_fraction - f]
        raw_data_low = np.memmap(FILE_LOW, dtype=np.uint8, mode="r").reshape(-1, 15)
        if use_fraction < 1.0:
            n_samples = int(len(raw_data_low) * fractions[0])
            idxs = np.random.choice(len(raw_data_low), n_samples, replace=False)
            raw_data_low = raw_data_low[idxs]
        raw_data_high = np.memmap(FILE_HIGH, dtype=np.uint8, mode="r").reshape(-1, 15)
        if use_fraction < 1.0:
            n_samples = int(len(raw_data_high) * fractions[1])
            idxs = np.random.choice(len(raw_data_high), n_samples, replace=False)
            raw_data_high = raw_data_high[idxs]

        # Select exposures to use
        raw_data: np.ndarray
        match use_exposures:
            case "high":
                raw_data = raw_data_high
                del raw_data_low
            case "low":
                raw_data = raw_data_low
                del raw_data_high
            case "both":
                raw_data = np.concatenate((raw_data_low, raw_data_high), axis=0)

        data_array = np.zeros_like(raw_data, dtype=np.float32)
        n_rows = raw_data.shape[0]

        # Convert each set of BGR values to HSI in chunks
        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_rows)
            chunk_bgr = raw_data[start_idx:end_idx]

            hsi_data_list = []
            for i in range(0, chunk_bgr.shape[1], 3):
                bgr_img = chunk_bgr[:, i : i + 3].reshape(-1, 1, 3)
                hsi_img = bm.BGR2HSI(bgr_img)
                hsi_data_list.append(hsi_img.reshape(-1, 3))

            data_array[start_idx:end_idx] = np.concatenate(hsi_data_list, axis=1)

        # Shuffle data_array
        if pre_shuffle:
            np.random.shuffle(data_array)

        self.data_array = data_array

        self.shuffle_batch = shuffle_batch
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.data_array) // self.batch_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.batch_size
        end = start + self.batch_size

        batch_data = self.data_array[start:end]

        # Shuffle batch_data
        if self.shuffle_batch:
            np.random.shuffle(batch_data)

        inputs = torch.tensor(batch_data[:, :12], dtype=torch.float32)
        outputs = torch.tensor(batch_data[:, 12:], dtype=torch.float32)

        return inputs, outputs
