#!/usr/bin/env python3
"""
한국어 띄어쓰기 및 개행 예측을 위한 문자 단위 경량 모델 학습 스크립트
허깅페이스 한글 데이터셋을 기반으로 학습 데이터를 구성합니다.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACT_DIR = MODELS_DIR / "korean_spacing_char"
MODEL_STATE_PATH = MODEL_ARTIFACT_DIR / "char_spacing_model.pt"
MODEL_CONFIG_PATH = MODEL_ARTIFACT_DIR / "config.json"

MAX_LENGTH = 512
NUM_LABELS = 3  # 0: 없음, 1: 공백, 2: 개행
CHAR_VOCAB_SIZE = 65536
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = CHAR_VOCAB_SIZE - 1
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", 32))
EPOCHS = int(os.getenv("TRAIN_EPOCHS", 3))
LEARNING_RATE = float(os.getenv("TRAIN_LR", 3e-4))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", 60000))
EVAL_SPLIT = float(os.getenv("EVAL_SPLIT", 0.1))
SEED = int(os.getenv("TRAIN_SEED", 42))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_CANDIDATES_ENV = os.getenv("DATASET_CANDIDATES")
DATASET_MODE = os.getenv("DATASET_MODE", "fallback").lower()
DEFAULT_DATASET_CANDIDATES = [
    ("klue", "ynat"),
    ("klue", "nli"),
    ("klue", "sts"),
    ("lcw99/wikipedia-korean-20221001", None),
    ("HAERAE-HUB/KOREAN-WEBTEXT", None),
]

CURRICULUM_ENABLED = os.getenv("CURRICULUM_ENABLED", "true").lower() != "false"
CURRICULUM_SCHEDULE_ENV = os.getenv("CURRICULUM_SCHEDULE")
DEFAULT_CURRICULUM = [
    {"name": "short", "max_length": 80, "epochs": 1},
    {"name": "medium", "max_length": 160, "epochs": 1},
    {"name": "full", "max_length": MAX_LENGTH, "epochs": max(1, EPOCHS)},
]

AMP_ENABLED_ENV = os.getenv("AMP_ENABLED")
if AMP_ENABLED_ENV is None:
    AMP_ENABLED = torch.cuda.is_available()
else:
    AMP_ENABLED = AMP_ENABLED_ENV.lower() == "true"

GRAD_CLIP_ENABLED = os.getenv("GRAD_CLIP_ENABLED", "true").lower() != "false"
GRAD_CLIP_MAX_NORM = float(os.getenv("GRAD_CLIP_MAX_NORM", "1.0"))
BOUNDARY_EPS = 1e-8
CONTINUE_TRAIN = os.getenv("CONTINUE_TRAIN", "true").lower() == "true"

os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def char_to_id(ch: str) -> int:
    """문자를 정수 ID로 변환 (0은 PAD 전용, 미지 문자는 최댓값으로 매핑)."""
    code = ord(ch) + 1  # 0은 패딩 용도로 예약
    if code >= CHAR_VOCAB_SIZE:
        return UNK_TOKEN_ID
    return code


class SpacingDataset(Dataset):
    """문자 단위 입력과 공백/개행 라벨을 제공하는 Dataset."""

    def __init__(
        self, texts: List[str], labels: List[List[int]], max_length: int = MAX_LENGTH
    ):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label_seq = self.labels[idx]

        input_ids = torch.full((self.max_length,), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        label_tensor = torch.full((self.max_length,), -100, dtype=torch.long)

        chars = [char_to_id(ch) for ch in text][: self.max_length]
        labels = label_seq[: self.max_length]
        length = len(chars)

        input_ids[:length] = torch.tensor(chars, dtype=torch.long)
        attention_mask[:length] = 1
        label_tensor[:length] = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "length": length,
        }


class CharSpacingModel(nn.Module):
    """간단한 BiLSTM 기반 공백/개행 예측 모델."""

    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=PAD_TOKEN_ID
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
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        x = self.embedding(input_ids)
        encoded, _ = self.encoder(x)
        encoded = self.dropout(encoded)
        logits = self.classifier(encoded)
        return logits


def parse_dataset_candidates():
    """환경 변수 또는 기본 설정에서 데이터셋 후보 목록 생성."""
    if DATASET_CANDIDATES_ENV:
        candidates: List[Tuple[str, Optional[str]]] = []
        for item in DATASET_CANDIDATES_ENV.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                name, config = item.split(":", 1)
                candidates.append((name.strip(), config.strip()))
            else:
                candidates.append((item, None))
        if candidates:
            return candidates
    return DEFAULT_DATASET_CANDIDATES


def parse_curriculum_schedule():
    """환경 변수에서 커리큘럼 구성을 파싱."""
    if not CURRICULUM_ENABLED:
        return [{"name": "full", "max_length": MAX_LENGTH, "epochs": EPOCHS}]

    if CURRICULUM_SCHEDULE_ENV:
        schedule = []
        for idx, raw in enumerate(CURRICULUM_SCHEDULE_ENV.split(",")):
            raw = raw.strip()
            if not raw:
                continue
            if ":" in raw:
                max_len_str, epochs_str = raw.split(":", 1)
            else:
                max_len_str, epochs_str = raw, str(EPOCHS)
            try:
                max_len = int(max_len_str)
                epochs = int(epochs_str)
            except ValueError:
                continue
            if max_len <= 0 or epochs <= 0:
                continue
            schedule.append(
                {
                    "name": f"stage_{idx + 1}",
                    "max_length": max_len,
                    "epochs": epochs,
                }
            )
        if schedule:
            schedule.sort(key=lambda x: x["max_length"])
            return schedule

    return DEFAULT_CURRICULUM[:]


def load_korean_dataset():
    """허깅페이스에서 한국어 텍스트 데이터셋을 순차적으로 시도하며 로드."""
    datasets_to_try = parse_dataset_candidates()
    loaded = []

    for dataset_name, config in datasets_to_try:
        try:
            dataset = (
                load_dataset(dataset_name, config)
                if config
                else load_dataset(dataset_name)
            )
            print(f"[데이터셋] {dataset_name} (config: {config}) 로드 성공")
            loaded.append((dataset, dataset_name))
            if DATASET_MODE != "concat":
                break
        except Exception as exc:
            print(f"[경고] {dataset_name} 로드 실패: {exc}")

    if not loaded:
        print("[오류] 사용 가능한 한국어 데이터셋을 찾을 수 없습니다.")
        return None, None

    if DATASET_MODE == "concat" and len(loaded) > 1:
        print(f"[데이터셋] {len(loaded)}개 데이터셋을 병합합니다.")
        merged_dataset = loaded[0][0]
        merged_name = loaded[0][1]
        for dataset, dataset_name in loaded[1:]:
            merged_dataset = merged_dataset.concatenate(dataset)
            merged_name += f"+{dataset_name}"
        return merged_dataset, merged_name

    return loaded[0]


def extract_texts(example: dict, dataset_name: str) -> List[str]:
    """데이터셋 특정 필드 + 일반 필드에서 텍스트를 추출."""
    fields: List[str] = []
    dataset_name_lower = (dataset_name or "").lower()
    if dataset_name_lower.endswith("nsmc"):
        if isinstance(example.get("document"), str):
            fields.append(example["document"])
    elif dataset_name_lower.startswith("klue"):
        for key in ("title", "sentence1", "sentence2", "premise", "hypothesis"):
            if isinstance(example.get(key), str):
                fields.append(example[key])
    elif "korean_wikipedia" in dataset_name_lower:
        for key in ("text", "content"):
            if isinstance(example.get(key), str):
                fields.append(example[key])
    elif "mc4" in dataset_name_lower:
        if isinstance(example.get("text"), str):
            fields.append(example["text"])
    elif "modu_corpus" in dataset_name_lower:
        for key in ("text", "body", "sentence", "utterance"):
            if isinstance(example.get(key), str):
                fields.append(example[key])
    elif "aihub" in dataset_name_lower:
        for key in ("text", "content", "utterance", "dialog"):
            value = example.get(key)
            if isinstance(value, str):
                fields.append(value)
            elif isinstance(value, list):
                fields.extend([item for item in value if isinstance(item, str)])

    for key in ("text", "document", "sentence", "context", "question", "answer"):
        value = example.get(key)
        if isinstance(value, str):
            fields.append(value)
        elif isinstance(value, list):
            fields.extend([item for item in value if isinstance(item, str)])

    # 중복 제거
    unique = []
    seen = set()
    for field in fields:
        if field not in seen:
            unique.append(field)
            seen.add(field)
    return unique


def build_example(text: str) -> Optional[Tuple[str, List[int]]]:
    """원본 텍스트에서 공백/개행 라벨과 문자 시퀀스를 생성."""
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    if len(text) < 5:
        return None

    base_chars: List[str] = []
    labels: List[int] = []

    for ch in text:
        if ch in {"\n", "\r"}:
            if labels:
                labels[-1] = 2  # 개행
            continue
        if ch.isspace():
            if labels and labels[-1] < 2:
                labels[-1] = 1  # 공백
            continue
        base_chars.append(ch)
        labels.append(0)

    if not base_chars:
        return None

    if len(base_chars) > MAX_LENGTH:
        base_chars = base_chars[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return "".join(base_chars), labels


def build_training_data(dataset, dataset_name: str):
    """데이터셋으로부터 학습/검증 데이터를 생성."""
    texts: List[str] = []
    labels: List[List[int]] = []

    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            continue

        split = dataset[split_name]
        print(f"[데이터 처리] {split_name} 분할 ({len(split)}개) ...")
        for example in tqdm(split, desc=f"{split_name} 처리"):
            for raw_text in extract_texts(example, dataset_name):
                example_data = build_example(raw_text)
                if example_data is None:
                    continue

                text, label_seq = example_data
                if len(text) < 3:
                    continue

                texts.append(text)
                labels.append(label_seq)

                if len(texts) >= MAX_SAMPLES:
                    break
            if len(texts) >= MAX_SAMPLES:
                break

        if len(texts) >= MAX_SAMPLES:
            print(f"[안내] MAX_SAMPLES={MAX_SAMPLES}에 도달하여 수집 종료")
            break

    if not texts:
        raise ValueError("학습에 사용할 텍스트를 생성하지 못했습니다.")

    print(f"[데이터] 총 {len(texts)}개 문장을 수집했습니다.")
    return texts, labels


def split_dataset(
    texts: List[str], labels: List[List[int]], eval_split: float = EVAL_SPLIT
):
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - eval_split))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    def subset(idx_list):
        return [texts[i] for i in idx_list], [labels[i] for i in idx_list]

    return subset(train_idx), subset(val_idx)


def evaluate(model: CharSpacingModel, data_loader: DataLoader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    tp = fp = fn = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(
                device_type=AMP_DEVICE_TYPE,
                enabled=AMP_ENABLED and AMP_DEVICE_TYPE == "cuda",
            ):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            labels_flat = labels.view(-1)
            preds_flat = preds.view(-1)
            mask = labels_flat != -100

            correct += (preds_flat[mask] == labels_flat[mask]).sum().item()
            total_tokens += mask.sum().item()

            gold_boundary = labels_flat[mask] > 0
            pred_boundary = preds_flat[mask] > 0
            tp += torch.logical_and(gold_boundary, pred_boundary).sum().item()
            fp += torch.logical_and(~gold_boundary, pred_boundary).sum().item()
            fn += torch.logical_and(gold_boundary, ~pred_boundary).sum().item()

    avg_loss = total_loss / max(1, len(data_loader))
    accuracy = correct / max(1, total_tokens)

    precision = tp / (tp + fp + BOUNDARY_EPS)
    recall = tp / (tp + fn + BOUNDARY_EPS)
    boundary_f1 = 2 * precision * recall / (precision + recall + BOUNDARY_EPS)

    return avg_loss, accuracy, boundary_f1


def train_model():
    dataset, dataset_name = load_korean_dataset()
    if dataset is None:
        raise RuntimeError("허깅페이스 한국어 데이터셋을 로드하지 못했습니다.")

    texts, labels = build_training_data(dataset, dataset_name)
    dataset_pairs = list(zip(texts, labels))
    schedule = parse_curriculum_schedule()

    model = CharSpacingModel().to(DEVICE)
    if CONTINUE_TRAIN and MODEL_STATE_PATH.exists():
        state_dict = torch.load(MODEL_STATE_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"[모델] 기존 체크포인트를 로드했습니다: {MODEL_STATE_PATH}")
    elif CONTINUE_TRAIN:
        print(f"[모델] CONTINUE_TRAIN=true 이지만 체크포인트가 없어 새로 학습합니다.")
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=AMP_ENABLED)

    best_val_accuracy = 0.0
    lower_bound = 0
    for stage_idx, stage in enumerate(schedule, start=1):
        stage_name = stage.get("name", f"stage_{stage_idx}")
        stage_max = min(stage.get("max_length", MAX_LENGTH), MAX_LENGTH)
        stage_epochs = max(1, stage.get("epochs", EPOCHS))
        stage_min = stage.get("min_length", lower_bound)

        stage_pairs = [
            (text, label)
            for text, label in dataset_pairs
            if stage_min < len(text) <= stage_max
        ]

        if not stage_pairs:
            print(f"[커리큘럼] {stage_name} 단계에 해당하는 데이터가 없어 건너뜁니다.")
            lower_bound = stage_max
            continue

        stage_texts, stage_labels = zip(*stage_pairs)
        (train_texts, train_labels), (val_texts, val_labels) = split_dataset(
            list(stage_texts), list(stage_labels)
        )
        print(
            f"[커리큘럼] {stage_name} 단계 - 길이 ({stage_min}, {stage_max}] "
            f"훈련 {len(train_texts)}개 / 검증 {len(val_texts)}개, epochs={stage_epochs}"
        )

        train_dataset = SpacingDataset(train_texts, train_labels)
        val_dataset = SpacingDataset(val_texts, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        for epoch in range(1, stage_epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in tqdm(
                train_loader, desc=f"[학습:{stage_name}] Epoch {epoch}/{stage_epochs}"
            ):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                label_tensor = batch["labels"].to(DEVICE)

                with torch.amp.autocast(
                    device_type=AMP_DEVICE_TYPE,
                    enabled=AMP_ENABLED and AMP_DEVICE_TYPE == "cuda",
                ):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits.view(-1, NUM_LABELS), label_tensor.view(-1))

                if AMP_ENABLED:
                    scaler.scale(loss).backward()
                    if GRAD_CLIP_ENABLED:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if GRAD_CLIP_ENABLED:
                        clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                    optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / max(1, len(train_loader))
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
            print(
                f"[{stage_name} | Epoch {epoch}] "
                f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), MODEL_STATE_PATH)
                print(f"[모델 저장] {MODEL_STATE_PATH} (val_acc={val_acc:.4f})")

        lower_bound = stage_max

    config = {
        "max_length": MAX_LENGTH,
        "num_labels": NUM_LABELS,
        "char_vocab_size": CHAR_VOCAB_SIZE,
        "pad_token_id": PAD_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.2,
    }
    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)
    print(f"[구성 저장] {MODEL_CONFIG_PATH}")
    print("[완료] 학습이 끝났습니다.")


if __name__ == "__main__":
    train_model()
