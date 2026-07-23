"""Decoder-only GPT (Radford et al. 2018 / Brown et al. 2020).

Ported from CAP5636_W6_Transformer(LLM).ipynb (Module 1 + the Module 3 loss
convention). Kept dependency-injectable (pos_encoder / block_class / norm_class
/ activation) so Stage 1 and Stage 2 share one model definition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# HF/SFT convention: masked label positions are ignored by cross-entropy.
IGNORE_INDEX = -100


@dataclass
class GPTConfig:
    vocab_size: int = 8_000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    d_ff: Optional[int] = None
    dropout: float = 0.10


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return self.dropout(x + self.emb(pos))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, V)
        if return_weights:
            return output, weights
        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
        max_len: int = 2048,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_drop = nn.Dropout(out_dropout)
        self.attn = ScaledDotProductAttention(dropout=attn_dropout)

        if causal:
            mask = torch.full((max_len, max_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("causal_mask", mask)
        else:
            self.causal_mask = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        B, T, _ = x.shape
        H, D = self.n_head, self.head_dim

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        mask = None
        if self.causal_mask is not None:
            mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            pad = key_padding_mask.float().masked_fill(key_padding_mask, float("-inf")).view(B, 1, 1, T)
            mask = pad if mask is None else mask + pad

        out, weights = self.attn(q, k, v, mask=mask, return_weights=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_drop(self.c_proj(out))

        if return_weights:
            return out, weights
        return out


class PositionwiseFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        activation = activation or nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        norm_class: Type[nn.Module] = nn.LayerNorm,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.norm1 = norm_class(config.n_embd)
        self.attn = MultiHeadAttention(
            d_model=config.n_embd,
            n_head=config.n_head,
            attn_dropout=config.dropout,
            out_dropout=config.dropout,
            bias=False,
            causal=True,
            max_len=config.block_size,
        )
        self.norm2 = norm_class(config.n_embd)
        self.ffn = PositionwiseFFN(
            d_model=config.n_embd,
            d_ff=config.d_ff,
            activation=activation,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        if return_weights:
            attn_out, weights = self.attn(self.norm1(x), return_weights=True)
        else:
            attn_out = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if return_weights:
            return x, weights
        return x


class GPT(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        pos_encoder: nn.Module,
        block_class: Type[nn.Module] = GPTBlock,
        norm_class: Type[nn.Module] = nn.LayerNorm,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_enc = pos_encoder
        self.blocks = nn.ModuleList(
            [block_class(config, norm_class=norm_class, activation=activation) for _ in range(config.n_layer)]
        )
        self.final_norm = norm_class(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: saves V*d params, improves perplexity (Press & Wolf 2017).
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Input length {T} exceeds block_size {self.config.block_size}"
        x = self.tok_emb(idx)
        x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_all_attn_weights(self, idx: torch.Tensor) -> List[torch.Tensor]:
        x = self.pos_enc(self.tok_emb(idx))
        weights = []
        for block in self.blocks:
            x, w = block(x, return_weights=True)
            weights.append(w)
        return weights


def build_model(config: GPTConfig, pos_encoding: str = "learned") -> GPT:
    """Construct a GPT with the requested positional encoding ("learned" | "sinusoidal")."""
    if pos_encoding == "learned":
        pos_enc: nn.Module = LearnedPositionalEncoding(
            d_model=config.n_embd, max_len=config.block_size, dropout=config.dropout
        )
    elif pos_encoding == "sinusoidal":
        pos_enc = SinusoidalPositionalEncoding(
            d_model=config.n_embd, max_len=config.block_size, dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown pos_encoding: {pos_encoding!r}")
    return GPT(config=config, pos_encoder=pos_enc)
