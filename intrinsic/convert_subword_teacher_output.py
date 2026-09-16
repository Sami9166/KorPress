"""Convert subword teacher decisions into a row-wise gzipped CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 파싱 실패: {path}:{line_number}") from exc
            sentence_id = record.get("sentence_id")
            if not sentence_id or sentence_id in seen:
                raise ValueError(f"sentence_id가 없거나 중복됩니다: {path}:{line_number}")
            seen.add(sentence_id)
            records.append(record)
    if not records:
        raise ValueError(f"JSONL이 비어 있습니다: {path}")
    return records


def _validate_input(records: list[dict]) -> dict[str, dict]:
    by_id = {}
    for record in records:
        sentence_id = record["sentence_id"]
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"tokens가 비어 있거나 list가 아닙니다: {sentence_id}")
        for index, token in enumerate(tokens):
            if token.get("index") != index:
                raise ValueError(f"token index가 연속적이지 않습니다: {sentence_id}[{index}]")
            if "token" not in token or "word_index" not in token:
                raise ValueError(f"token 필드가 부족합니다: {sentence_id}[{index}]")
        by_id[sentence_id] = record
    return by_id


def _validate_teacher_output(records: list[dict], input_by_id: dict[str, dict]) -> dict[str, set[int]]:
    decisions: dict[str, set[int]] = {}
    for record in records:
        sentence_id = record.get("sentence_id")
        if sentence_id not in input_by_id:
            raise ValueError(f"입력에 없는 sentence_id입니다: {sentence_id}")
        raw_indices = record.get("keep_indices")
        if not isinstance(raw_indices, list) or any(type(index) is not int for index in raw_indices):
            raise ValueError(f"keep_indices가 정수 list가 아닙니다: {sentence_id}")
        if raw_indices != sorted(set(raw_indices)):
            raise ValueError(f"keep_indices가 정렬되지 않았거나 중복됩니다: {sentence_id}")
        token_count = len(input_by_id[sentence_id]["tokens"])
        if any(index < 0 or index >= token_count for index in raw_indices):
            raise ValueError(f"keep_indices 범위를 벗어났습니다: {sentence_id}")
        decisions[sentence_id] = set(raw_indices)
    missing = [sentence_id for sentence_id in input_by_id if sentence_id not in decisions]
    if missing:
        raise ValueError(f"teacher 출력에 없는 sentence_id입니다: {missing[0]}")
    return decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-input", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_records = _read_jsonl(args.teacher_input)
    output_records = _read_jsonl(args.teacher_output)
    input_by_id = _validate_input(input_records)
    decisions = _validate_teacher_output(output_records, input_by_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sentence_id",
        "token_index",
        "subword_index",
        "word_index",
        "token",
        "token_id",
        "label",
    ]
    rows = 0
    kept = 0
    with gzip.open(args.output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in input_records:
            sentence_id = record["sentence_id"]
            keep_indices = decisions[sentence_id]
            for index, token in enumerate(record["tokens"]):
                is_kept = index in keep_indices
                writer.writerow(
                    {
                        "sentence_id": sentence_id,
                        "token_index": index,
                        "subword_index": index,
                        "word_index": token["word_index"],
                        "token": token["token"],
                        "token_id": token.get("token_id", ""),
                        "label": "PRESERVE" if is_kept else "DISCARD",
                    }
                )
                rows += 1
                kept += is_kept
    print(
        json.dumps(
            {
                "sentences": len(input_records),
                "subwords": rows,
                "preserve": kept,
                "discard": rows - kept,
                "preserve_rate": kept / rows,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
