import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        b, t, n = x.shape
        x = x.permute(0, 2, 1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patch_emb = self.proj(patches)
        return patch_emb, patch_emb.size(2)