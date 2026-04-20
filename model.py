import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config import ModelConfig
from modules.revin import RevIN
from modules.patch import PatchEmbedding
from modules.informer_encoder import InformerEncoder
from modules.prototypes import TextPrototypeBank
from modules.cpp import CPPEncoder


class InformerLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.revin = RevIN(cfg.num_vars, eps=cfg.revin_eps, affine=cfg.revin_affine)
        self.patch_embed = PatchEmbedding(cfg.patch_len, cfg.stride, cfg.d_model)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.gpt_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModel.from_pretrained(cfg.gpt_name)
        self.llm_dim = self.llm.config.hidden_size

        if cfg.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        llm_embed_weight = self.llm.get_input_embeddings().weight
        self.proto_bank = TextPrototypeBank(cfg.proto_len, llm_embed_weight, cfg.d_model)

        self.informer = InformerEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            e_layers=cfg.e_layers,
            dropout=cfg.dropout,
            factor=cfg.factor,
            activation=cfg.activation
        )

        self.align_to_llm = nn.Linear(cfg.d_model, self.llm_dim)
        self.cpp_encoder = CPPEncoder(self.tokenizer, self.llm, max_len=cfg.cpp_max_len)

        self.temporal_token_count = self._infer_temporal_token_count(cfg.seq_len)
        self.pred_head = nn.Linear(self.temporal_token_count * self.llm_dim, cfg.pred_len)

    def _infer_temporal_token_count(self, seq_len):
        P = (seq_len - self.cfg.patch_len) // self.cfg.stride + 1
        L = P + self.cfg.proto_len
        for _ in range(self.cfg.e_layers - 1):
            L = math.floor((L + 1) / 2)
        return L

    def build_cpp_prompts(self, x_norm):
        B, T, N = x_norm.shape
        prompts = []

        for b in range(B):
            sample = x_norm[b]
            trend = sample[-1].mean().item() - sample[0].mean().item()
            volatility = sample.std().item()

            trend_desc = "upward trend" if trend > 0.05 else ("downward trend" if trend < -0.05 else "stable trend")
            vol_desc = "high volatility" if volatility > 1.0 else "moderate volatility"

            prompt = (
                f"This is a multivariate time series forecasting dataset. "
                f"The task is to predict the next {self.cfg.pred_len} time steps from {T} historical observations. "
                f"The input contains {N} variables. "
                f"The sequence shows {trend_desc} and {vol_desc}."
            )
            prompts.append(prompt)

        return prompts

    def forward(self, x):
        B, T, N = x.shape
        device = x.device

        x_norm = self.revin(x, mode="norm")
        patch_emb, P = self.patch_embed(x_norm)
        patch_emb = patch_emb.reshape(B * N, P, self.cfg.d_model)

        proto = self.proto_bank(B * N)
        z = torch.cat([patch_emb, proto], dim=1)

        z = self.informer(z)
        z_llm = self.align_to_llm(z)
        L_ts = z_llm.size(1)

        prompts = self.build_cpp_prompts(x_norm)
        prompt_emb = self.cpp_encoder(prompts, device=device)
        L_prompt = prompt_emb.size(1)

        prompt_emb = prompt_emb.unsqueeze(1).expand(B, N, L_prompt, self.llm_dim)
        prompt_emb = prompt_emb.reshape(B * N, L_prompt, self.llm_dim)

        llm_inputs = torch.cat([prompt_emb, z_llm], dim=1)
        llm_out = self.llm(inputs_embeds=llm_inputs).last_hidden_state

        temporal_out = llm_out[:, L_prompt:, :]
        temporal_flat = temporal_out.reshape(B * N, -1)
        pred = self.pred_head(temporal_flat)

        pred = pred.view(B, N, self.cfg.pred_len).permute(0, 2, 1)
        pred = self.revin(pred, mode="denorm")
        return pred