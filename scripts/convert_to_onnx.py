"""
문자 단위 띄어쓰기/개행 모델을 ONNX로 변환합니다.
"""

from __future__ import annotations

import os
import sys

os.environ["ONNX_DISABLE_OPTIMIZER"] = "1"
os.environ["ONNX_DISABLE_CONSTANT_FOLDING"] = "1"
os.environ["TORCH_ONNX_DISABLE_OPTIMIZER"] = "1"

import json
from pathlib import Path

import torch
import onnx
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACT_DIR = MODELS_DIR / "korean_spacing_char"
MODEL_STATE_PATH = MODEL_ARTIFACT_DIR / "char_spacing_model.pt"
MODEL_CONFIG_PATH = MODEL_ARTIFACT_DIR / "config.json"
ONNX_OUTPUT_PATH = MODELS_DIR / "korean_spacing.onnx"


class CharSpacingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout: float,
        pad_token_id: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        x = self.embedding(input_ids)
        encoded, _ = self.encoder(x)
        encoded = self.dropout(encoded)
        logits = self.classifier(encoded)
        return logits


def convert_to_onnx():
    if not MODEL_STATE_PATH.exists() or not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "훈련된 모델이 없습니다. 먼저 scripts/train.py를 실행하여 "
            "char_spacing_model.pt 및 config.json을 생성하세요."
        )

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    model = CharSpacingModel(
        vocab_size=config["char_vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_labels=config["num_labels"],
        dropout=config["dropout"],
        pad_token_id=config["pad_token_id"],
    )
    state_dict = torch.load(MODEL_STATE_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input_ids = torch.zeros(1, config["max_length"], dtype=torch.long)
    dummy_attention_mask = torch.ones(1, config["max_length"], dtype=torch.long)

    print("[ONNX] 변환 시작...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        ONNX_OUTPUT_PATH.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"[ONNX] 변환 완료: {ONNX_OUTPUT_PATH}")

    # onnx
    # _model = onnx.load(ONNX_OUTPUT_PATH.as_posix())
    # onnx.checker.check_model(onnx_model)
    print("[ONNX] 변환 완료 (검증 스킵)")
    sys.exit(0)


if __name__ == "__main__":
    convert_to_onnx()
