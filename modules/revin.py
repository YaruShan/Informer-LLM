import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

        self._mean = None
        self._std = None

    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
            return x

        if mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self._std + self._mean
            return x

        raise ValueError("mode must be 'norm' or 'denorm'")