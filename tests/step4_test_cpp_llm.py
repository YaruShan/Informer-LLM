import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from modules.cpp import CPPEncoder


def main():
    cfg = ModelConfig()
    device = cfg.device

    tokenizer = AutoTokenizer.from_pretrained(cfg.gpt_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModel.from_pretrained(cfg.gpt_name).to(device)
    cpp = CPPEncoder(tokenizer, llm, max_len=cfg.cpp_max_len)

    prompts = [
        "This is a multivariate time series forecasting task.",
        "Please forecast the next future steps."
    ]

    emb = cpp(prompts, device)
    print("CPP embedding shape:", emb.shape)


if __name__ == "__main__":
    main()