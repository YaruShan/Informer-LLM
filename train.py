import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig
from dataset import ETTHDataset
from model import InformerLLM


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = F.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        mse = F.mse_loss(pred, batch_y)
        mae = F.l1_loss(pred, batch_y)

        total_mse += mse.item()
        total_mae += mae.item()

    return total_mse / len(loader), total_mae / len(loader)


def main():
    cfg = ModelConfig()
    device = cfg.device

    print("Using device:", device)
    print("CSV path:", cfg.csv_path)

    train_set = ETTHDataset(cfg.csv_path, cfg.seq_len, cfg.pred_len, split="train")
    val_set = ETTHDataset(cfg.csv_path, cfg.seq_len, cfg.pred_len, split="val")
    test_set = ETTHDataset(cfg.csv_path, cfg.seq_len, cfg.pred_len, split="test")

    cfg.num_vars = train_set.num_vars

    print("Detected num_vars:", cfg.num_vars)
    print("Train samples:", len(train_set))
    print("Val samples  :", len(val_set))
    print("Test samples :", len(test_set))

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    model = InformerLLM(cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params    : {total_params / 1e6:.2f} M")
    print(f"Trainable params: {train_params / 1e6:.2f} M")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")

    x, y = next(iter(test_loader))
    x = x.to(device)
    pred = model(x)
    print("Input shape :", x.shape)
    print("Pred shape  :", pred.shape)


if __name__ == "__main__":
    main()