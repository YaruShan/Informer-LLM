import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from modules.revin import RevIN
from modules.patch import PatchEmbedding


def main():
    cfg = ModelConfig()

    x = torch.randn(2, cfg.seq_len, cfg.num_vars)
    revin = RevIN(cfg.num_vars)
    patch = PatchEmbedding(cfg.patch_len, cfg.stride, cfg.d_model)

    x_norm = revin(x, mode="norm")
    patch_emb, P = patch(x_norm)

    print("x_norm shape   :", x_norm.shape)
    print("patch_emb shape:", patch_emb.shape)
    print("num_patches    :", P)


if __name__ == "__main__":
    main()