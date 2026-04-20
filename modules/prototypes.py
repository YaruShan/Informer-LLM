import torch
import torch.nn as nn


class TextPrototypeBank(nn.Module):
    def __init__(self, proto_len, llm_embed_weight, d_model):
        super().__init__()
        init_proto = llm_embed_weight[:proto_len].detach().clone()
        self.proto = nn.Parameter(init_proto)
        self.proj = nn.Linear(llm_embed_weight.shape[1], d_model)

    def forward(self, batch_size):
        proto = self.proj(self.proto)
        return proto.unsqueeze(0).expand(batch_size, -1, -1)