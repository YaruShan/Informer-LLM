import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from dataset import ETTHDataset


def main():
    cfg = ModelConfig()
    ds = ETTHDataset(cfg.csv_path, cfg.seq_len, cfg.pred_len, split="train")
    x, y = ds[0]

    print("Dataset loaded successfully.")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("num_vars:", ds.num_vars)
    print("num_samples:", len(ds))


if __name__ == "__main__":
    main()