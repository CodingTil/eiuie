from typing import Tuple, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
import base_model as bm

FILE_LOW = "data/pixel_dataset/low.ds"
FILE_HIGH = "data/pixel_dataset/high.ds"


class PixelDataset(Dataset):
    def __init__(
        self,
        chunk_size=10000,
        use_fraction=1.0,
        use_exposures: Literal["high", "low", "both"] = "both",
    ):
        # Ensure use_fraction is within valid bounds
        use_fraction = max(0.0, min(1.0, use_fraction))

        # Use numpy's memory mapping
        raw_data_low = np.memmap(FILE_LOW, dtype=np.uint8, mode="r").reshape(-1, 15)
        raw_data_high = np.memmap(FILE_HIGH, dtype=np.uint8, mode="r").reshape(-1, 15)

        # Select exposures to use
        raw_data: np.ndarray
        match use_exposures:
            case "high":
                raw_data = raw_data_high
            case "low":
                raw_data = raw_data_low
            case "both":
                raw_data = np.concatenate((raw_data_low, raw_data_high), axis=0)

        # Randomly select a fraction of the data if use_fraction < 1.0
        if use_fraction < 1.0:
            n_samples = int(len(raw_data) * use_fraction)
            idxs = np.random.choice(len(raw_data), n_samples, replace=False)
            raw_data = raw_data[idxs]

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
        np.random.shuffle(data_array)

        self.data_array = torch.from_numpy(data_array)

    def __len__(self) -> int:
        return len(self.data_array)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.data_array[index, :12]
        outputs = self.data_array[index, 12:]
        return inputs, outputs
