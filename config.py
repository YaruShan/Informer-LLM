from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    csv_path: str = r"C:\Users\肉肉\Desktop\ETTh1.csv"

    seq_len: int = 96
    pred_len: int = 96
    num_vars: int = 7

    revin_eps: float = 1e-5
    revin_affine: bool = True

    patch_len: int = 16
    stride: int = 8

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    e_layers: int = 2
    dropout: float = 0.1
    factor: int = 5
    activation: str = "gelu"

    proto_len: int = 8
    cpp_max_len: int = 128

    gpt_name: str = "meta-llama/Llama-2-7b-hf"
    freeze_llm: bool = True

    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    epochs: int = 10

    device: str = "cuda"