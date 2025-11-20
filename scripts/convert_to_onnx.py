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
import onnx  # noqa: F401

from korean_spacing.modeling import CharSpacingModel


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACT_DIR = MODELS_DIR / "korean_spacing_char"
MODEL_STATE_PATH = MODEL_ARTIFACT_DIR / "char_spacing_model.pt"
MODEL_CONFIG_PATH = MODEL_ARTIFACT_DIR / "config.json"
ONNX_OUTPUT_PATH = MODELS_DIR / "korean_spacing.onnx"


def convert_to_onnx():
    if not MODEL_STATE_PATH.exists() or not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "훈련된 모델이 없습니다. 먼저 scripts/train.py를 실행하여 "
            "char_spacing_model.pt 및 config.json을 생성하세요."
        )

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    model = CharSpacingModel(
        char_vocab_size=config["char_vocab_size"],
        num_labels=config["num_labels"],
        pad_token_id=config["pad_token_id"],
        max_length=config["max_length"],
        char_embedding_dim=config["char_embedding_dim"],
        subchar_embedding_dim=config["subchar_embedding_dim"],
        model_dim=config["model_dim"],
        num_heads=config["transformer_heads"],
        ff_dim=config["transformer_feedforward_dim"],
        num_layers=config["transformer_layers"],
        dropout=config["transformer_dropout"],
        choseong_vocab_size=config["choseong_vocab_size"],
        jungseong_vocab_size=config["jungseong_vocab_size"],
        jongseong_vocab_size=config["jongseong_vocab_size"],
    )
    state_dict = torch.load(MODEL_STATE_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input_ids = torch.zeros(1, config["max_length"], dtype=torch.long)
    dummy_attention_mask = torch.ones(1, config["max_length"], dtype=torch.long)
    dummy_choseong_ids = torch.zeros(1, config["max_length"], dtype=torch.long)
    dummy_jungseong_ids = torch.zeros(1, config["max_length"], dtype=torch.long)
    dummy_jongseong_ids = torch.zeros(1, config["max_length"], dtype=torch.long)

    print("[ONNX] 변환 시작...")
    torch.onnx.export(
        model,
        (
            dummy_input_ids,
            dummy_attention_mask,
            dummy_choseong_ids,
            dummy_jungseong_ids,
            dummy_jongseong_ids,
        ),
        ONNX_OUTPUT_PATH.as_posix(),
        input_names=[
            "input_ids",
            "attention_mask",
            "choseong_ids",
            "jungseong_ids",
            "jongseong_ids",
        ],
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
