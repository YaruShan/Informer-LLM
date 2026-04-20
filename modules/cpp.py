import torch
import torch.nn as nn


class CPPEncoder(nn.Module):
    def __init__(self, tokenizer, llm_model, max_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.max_len = max_len

    def forward(self, prompts, device):
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        word_embed = self.llm_model.get_input_embeddings()(input_ids)
        word_embed = word_embed * attention_mask.unsqueeze(-1)
        return word_embed