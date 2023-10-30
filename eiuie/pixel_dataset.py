from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FILE = "data/pixel_dataset.ds"


class PixelDataset(Dataset):
    """
    PixelDataset class.

    Attributes
    ----------
    df: pd.DataFrame
        Dataframe.
    """

    df: pd.DataFrame

    def __init__(self):
        # Load binary data
        with open(FILE, "rb") as f:
            raw_data = f.read()

        # Convert binary data to a numpy array of shape (num_rows, 15)
        data_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 15)

        # Convert numpy array to pandas dataframe
        self.df = pd.DataFrame(data_array)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx].values

        # Splitting the 15 values into two tensors: first 12 and last 3.
        input_tensor = torch.tensor(row[:12], dtype=torch.float32)
        output_tensor = torch.tensor(row[12:], dtype=torch.float32)

        return input_tensor, output_tensor
