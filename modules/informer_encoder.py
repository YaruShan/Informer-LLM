import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        b, l, h, e = queries.shape
        _, s, _, d = values.shape
        scale = self.scale or 1. / sqrt(e)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = torch.softmax(scale * scores, dim=-1)
        attn = self.dropout(attn)
        v = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return v.contiguous(), attn
        return v.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        b, h, l_k, e = K.shape
        _, _, l_q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(b, h, l_q, l_k, e)
        index_sample = torch.randint(l_k, (l_q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(l_q, device=Q.device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), l_k)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[
            torch.arange(b, device=Q.device)[:, None, None],
            torch.arange(h, device=Q.device)[None, :, None],
            M_top,
            :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, l_q):
        b, h, l_v, d = V.shape
        V_sum = V.mean(dim=-2)
        context = V_sum.unsqueeze(-2).expand(b, h, l_q, V_sum.shape[-1]).clone()
        return context

    def _update_context(self, context_in, V, scores, index, l_q):
        b, h, l_v, d = V.shape
        attn = torch.softmax(scores, dim=-1)
        context_in[
            torch.arange(b, device=V.device)[:, None, None],
            torch.arange(h, device=V.device)[None, :, None],
            index,
            :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([b, h, l_v, l_v], device=attn.device) / l_v).type_as(attn)
            attns[
                torch.arange(b, device=V.device)[:, None, None],
                torch.arange(h, device=V.device)[None, :, None],
                index,
                :
            ] = attn
            return context_in, attns

        return context_in, None

    def forward(self, queries, keys, values, attn_mask=None):
        b, l_q, h, d = queries.shape
        _, l_k, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * int(np.ceil(np.log(l_k)))
        u = self.factor * int(np.ceil(np.log(l_q)))

        U_part = U_part if U_part < l_k else l_k
        u = u if u < l_q else l_q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(d)
        scores_top = scores_top * scale

        context = self._get_initial_context(values, l_q)
        context, attn = self._update_context(context, values, scores_top, index, l_q)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None):
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads

        queries = self.query_projection(queries).view(b, l, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        if self.mix:
            out = out.transpose(2, 1).contiguous()

        out = out.view(b, l, -1)
        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, e_layers, dropout, factor, activation):
        super().__init__()

        attn = ProbAttention(False, factor, attention_dropout=dropout, output_attention=False)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(attn, d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model)
                for _ in range(e_layers - 1)
            ] if e_layers > 1 else None,
            norm_layer=nn.LayerNorm(d_model)
        )

    def forward(self, x):
        x, _ = self.encoder(x)
        return x