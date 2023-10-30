from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import base_model as bm

FILE = "data/pixel_dataset.ds"


class PixelDataset(Dataset):
    def __init__(self, batch_size=1):
        # Use numpy's memory mapping
        raw_data = np.memmap(FILE, dtype=np.uint8, mode="r").reshape(-1, 15)

        # Convert each set of BGR values to HSI
        hsi_data_list = []
        for i in range(0, raw_data.shape[1], 3):
            bgr_img = raw_data[:, i : i + 3].reshape(-1, 1, 3)
            hsi_img = bm.BGR2HSI(bgr_img)
            hsi_data_list.append(hsi_img.reshape(-1, 3))

        self.data_array = np.concatenate(hsi_data_list, axis=1)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.data_array) // self.batch_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.batch_size
        end = start + self.batch_size

        batch_data = self.data_array[start:end]

        inputs = torch.tensor(batch_data[:, :12], dtype=torch.float32)
        outputs = torch.tensor(batch_data[:, 12:], dtype=torch.float32)

        return inputs, outputs
