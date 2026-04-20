import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ETTHDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        seq_len: int,
        pred_len: int,
        split: str = "train",
        split_ratio=(0.7, 0.1, 0.2)
    ):
        super().__init__()
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if "date" in df.columns:
            df = df.drop(columns=["date"])

        df = df.select_dtypes(include=[np.number])
        data = df.values.astype(np.float32)

        total_len = len(data)
        train_end = int(total_len * split_ratio[0])
        val_end = int(total_len * (split_ratio[0] + split_ratio[1]))

        if split == "train":
            split_data = data[:train_end]
        elif split == "val":
            split_data = data[train_end:val_end]
        elif split == "test":
            split_data = data[val_end:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.data = split_data
        self.num_vars = self.data.shape[1]
        self.length = len(self.data) - self.seq_len - self.pred_len + 1

        if self.length <= 0:
            raise ValueError(
                f"Not enough data in split={split}. "
                f"split length={len(self.data)}, "
                f"need at least seq_len + pred_len = {self.seq_len + self.pred_len}"
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)