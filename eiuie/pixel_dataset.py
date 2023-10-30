from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import base_model as bm

FILE = "data/pixel_dataset.ds"

class PixelDataset(Dataset):
    def __init__(self, batch_size=1, chunk_size=10000, use_fraction=1.0):
        # Ensure use_fraction is within valid bounds
        use_fraction = max(0.0, min(1.0, use_fraction))
        
        # Use numpy's memory mapping
        raw_data = np.memmap(FILE, dtype=np.uint8, mode="r").reshape(-1, 15)
        
        # Randomly select a fraction of the data if use_fraction < 1.0
        if use_fraction < 1.0:
            n_samples = int(len(raw_data) * use_fraction)
            idxs = np.random.choice(len(raw_data), n_samples, replace=False)
            raw_data = raw_data[idxs]
        
        self.data_array = np.zeros_like(raw_data, dtype=np.float32)
        n_rows = raw_data.shape[0]

        # Convert each set of BGR values to HSI in chunks
        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_rows)
            chunk_bgr = raw_data[start_idx:end_idx]
            
            hsi_data_list = []
            for i in range(0, chunk_bgr.shape[1], 3):
                bgr_img = chunk_bgr[:, i:i+3].reshape(-1, 1, 3)
                hsi_img = bm.BGR2HSI(bgr_img)
                hsi_data_list.append(hsi_img.reshape(-1, 3))
                
            self.data_array[start_idx:end_idx] = np.concatenate(hsi_data_list, axis=1)
            
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
