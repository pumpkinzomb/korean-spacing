from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class CharSpacingModel(nn.Module):
    """Transformer 기반 문자/자모 혼합 임베딩 띄어쓰기 모델."""

    def __init__(
        self,
        *,
        char_vocab_size: int,
        num_labels: int,
        pad_token_id: int,
        max_length: int,
        char_embedding_dim: int = 128,
        subchar_embedding_dim: int = 32,
        model_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        choseong_vocab_size: int,
        jungseong_vocab_size: int,
        jongseong_vocab_size: int,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=pad_token_id
        )
        self.choseong_embedding = nn.Embedding(
            choseong_vocab_size, subchar_embedding_dim, padding_idx=0
        )
        self.jungseong_embedding = nn.Embedding(
            jungseong_vocab_size, subchar_embedding_dim, padding_idx=0
        )
        self.jongseong_embedding = nn.Embedding(
            jongseong_vocab_size, subchar_embedding_dim, padding_idx=0
        )

        fused_dim = char_embedding_dim + (3 * subchar_embedding_dim)
        self.fuse_projection = nn.Linear(fused_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_embedding = nn.Embedding(max_length, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        choseong_ids: Optional[torch.Tensor] = None,
        jungseong_ids: Optional[torch.Tensor] = None,
        jongseong_ids: Optional[torch.Tensor] = None,
    ):
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        )
        choseong_ids = (
            choseong_ids if choseong_ids is not None else torch.zeros_like(input_ids)
        )
        jungseong_ids = (
            jungseong_ids if jungseong_ids is not None else torch.zeros_like(input_ids)
        )
        jongseong_ids = (
            jongseong_ids if jongseong_ids is not None else torch.zeros_like(input_ids)
        )

        char_embed = self.char_embedding(input_ids)
        cho_embed = self.choseong_embedding(choseong_ids)
        jung_embed = self.jungseong_embedding(jungseong_ids)
        jong_embed = self.jongseong_embedding(jongseong_ids)

        fused = torch.cat([char_embed, cho_embed, jung_embed, jong_embed], dim=-1)
        fused = self.fuse_projection(fused)

        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        fused = fused + self.positional_embedding(positions)
        fused = self.dropout(fused)

        key_padding_mask = attention_mask == 0
        encoded = self.encoder(fused, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        logits = self.classifier(encoded)
        return logits

