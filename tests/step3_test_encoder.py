import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from modules.informer_encoder import InformerEncoder


def main():
    cfg = ModelConfig()

    x = torch.randn(4, 20, cfg.d_model)
    encoder = InformerEncoder(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        e_layers=cfg.e_layers,
        dropout=cfg.dropout,
        factor=cfg.factor,
        activation=cfg.activation
    )

    y = encoder(x)
    print("encoder input shape :", x.shape)
    print("encoder output shape:", y.shape)


if __name__ == "__main__":
    main()