from typing import Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd

TSV_FILE = "data/pixel_dataset.tsv"


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
        self.df = pd.read_table(TSV_FILE, header=None)
        self.df = self.df.astype(float)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx].values

        # Splitting the 15 values into two tensors: first 12 and last 3.
        input_tensor = torch.tensor(row[:12], dtype=torch.float32)
        output_tensor = torch.tensor(row[12:], dtype=torch.float32)

        return input_tensor, output_tensor
