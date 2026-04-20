import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from model import InformerLLM


def main():
    cfg = ModelConfig()
    device = cfg.device

    model = InformerLLM(cfg).to(device)
    model.eval()

    x = torch.randn(2, cfg.seq_len, cfg.num_vars).to(device)

    with torch.no_grad():
        pred = model(x)

    print("Input shape:", x.shape)
    print("Pred shape :", pred.shape)


if __name__ == "__main__":
    main()