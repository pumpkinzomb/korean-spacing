#!/usr/bin/env python3
"""
모델 예측 라벨 시퀀스를 공백/개행이 포함된 문자열로 복원하는 후처리 스크립트.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, List, Sequence

SPACE_LABEL = 1
NEWLINE_LABEL = 2


def apply_spacing(base_text: str, label_ids: Sequence[int]) -> str:
    """
    원본 문자 시퀀스(base_text)에 라벨(0: 없음, 1: 공백, 2: 개행)을 적용해 문자열을 복원.
    """
    if not base_text:
        return base_text

    output: List[str] = []
    for idx, ch in enumerate(base_text):
        output.append(ch)
        if idx >= len(label_ids):
            continue
        label = label_ids[idx]
        if label == SPACE_LABEL:
            output.append(" ")
        elif label == NEWLINE_LABEL:
            output.append("\n")
    return "".join(output)


def apply_spacing_batch(
    base_texts: Sequence[str], label_batch: Iterable[Sequence[int]]
) -> List[str]:
    """
    여러 입력 문장/라벨 쌍을 한꺼번에 복원.
    """
    return [apply_spacing(text, labels) for text, labels in zip(base_texts, label_batch)]


def main():
    parser = argparse.ArgumentParser(
        description="라벨 시퀀스를 적용해 공백/개행이 포함된 문자열을 복원합니다.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="공백 제거된 원본 문자열 (학습 시 사용된 base text)",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="쉼표로 구분된 라벨 시퀀스 또는 JSON 배열 (예: 0,0,1,2,0)",
    )
    args = parser.parse_args()

    try:
        if args.labels.strip().startswith("["):
            labels = json.loads(args.labels)
        else:
            labels = [int(item.strip()) for item in args.labels.split(",") if item.strip()]
    except Exception as exc:  # pragma: no cover - 단순 CLI
        raise SystemExit(f"라벨 파싱 실패: {exc}")

    restored = apply_spacing(args.text, labels)
    print(restored)


if __name__ == "__main__":
    main()

