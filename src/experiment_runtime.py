"""Shared runtime helpers for the intrinsic and KorQuAD experiments.

The module contains only data loading, compression adapters, and metric
helpers.  It never starts a model by itself; the two experiment entry points
decide which models to load and when to write outputs.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:  # Running from a script with ``src`` on sys.path
    from compressor import compress
    from token_baseline import TokenBaselineCompressor
except ImportError:  # Running as an imported package
    from .compressor import compress
    from .token_baseline import TokenBaselineCompressor


NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:/-]\d+)*(?:\w*)")
DATE_RE = re.compile(
    r"\d{2,4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|"
    r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"
)
NEGATION_CUES = ("안", "못", "않", "없", "아니", "말지", "않다", "않는")


def open_text(path: Path, mode: str = "rt"):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8-sig", newline="")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 오류: {path}:{line_number}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open_text(path, "rt") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"저장할 행이 없습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_chunks(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    result: Dict[str, str] = {}
    for row in rows:
        sentence_id = row.get("sentence_id") or row.get("passage_id")
        text = row.get("sentence") or row.get("context")
        if not sentence_id or text is None:
            raise ValueError(f"chunks 파일에 sentence_id/sentence 열이 필요합니다: {path}")
        result[sentence_id] = text
    return result


def load_span_predictions(path: Path) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row in read_csv_rows(path):
        span_id = row.get("span_id")
        probability = row.get("drop_probability")
        if not span_id or probability in (None, ""):
            continue
        if span_id in result:
            raise ValueError(f"예측 파일에 중복 span_id가 있습니다: {span_id}")
        result[span_id] = float(probability)
    if not result:
        raise ValueError(f"drop_probability 예측을 읽지 못했습니다: {path}")
    return result


def _span_text(words: Sequence[str], indices: Sequence[int]) -> str:
    try:
        return " ".join(words[int(index)] for index in indices)
    except IndexError as exc:
        raise ValueError(f"span index가 chunk 길이를 벗어났습니다: {indices}") from exc


def build_spans_by_chunk(
    span_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    chunks: Mapping[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert encoder JSONL records to the compressor's 1-based word schema."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in span_records:
        sentence_id = str(record["sentence_id"])
        if sentence_id not in chunks:
            raise ValueError(f"span record의 sentence_id가 chunks에 없습니다: {sentence_id}")
        words = chunks[sentence_id].split()
        for raw_span in record.get("spans", []):
            span_id = str(raw_span["span_id"])
            if span_id in seen:
                raise ValueError(f"span records에 중복 span_id가 있습니다: {span_id}")
            if span_id not in predictions:
                raise KeyError(f"예측 파일에 span_id가 없습니다: {span_id}")
            indices = [int(index) for index in raw_span["eojeol_indices"]]
            seen.add(span_id)
            grouped[sentence_id].append(
                {
                    "span_id": span_id,
                    "size": int(raw_span.get("size", len(indices))),
                    "word_ids": [index + 1 for index in indices],
                    "text": _span_text(words, indices),
                    "p_drop": float(predictions[span_id]),
                }
            )
    return grouped


def compress_span_chunks(
    chunks: Mapping[str, str],
    spans_by_chunk: Mapping[str, Sequence[Mapping[str, Any]]],
    L: int,
    threshold: float,
    drop_rule: str,
    sentence_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    selected = sentence_ids if sentence_ids is not None else list(chunks)
    rows: List[Dict[str, Any]] = []
    for sentence_id in selected:
        original = chunks[sentence_id]
        words = [{"id": i + 1, "text": word} for i, word in enumerate(original.split())]
        result = compress(
            words,
            list(spans_by_chunk.get(sentence_id, [])),
            L=L,
            threshold=threshold,
            use_dummy=False,
            drop_rule=drop_rule,
        )
        result["sentence_id"] = sentence_id
        result["method"] = "Span"
        rows.append(result)
    return rows


def compress_token_chunks(
    chunks: Mapping[str, str],
    compressor: TokenBaselineCompressor,
    threshold: float,
    sentence_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Compress complete eojeols with the same-size token baseline."""
    selected = sentence_ids if sentence_ids is not None else list(chunks)
    rows: List[Dict[str, Any]] = []
    for sentence_id in selected:
        original = chunks[sentence_id]
        compressed = compressor.compress(original, threshold=threshold)
        rows.append(
            {
                "sentence_id": sentence_id,
                "method": "Token",
                "original": original,
                "compressed": compressed,
                "threshold": threshold,
                "n_words_original": len(original.split()),
                "n_words_compressed": len(compressed.split()),
                "n_words_dropped": len(original.split()) - len(compressed.split()),
            }
        )
    return rows


class LLMLingua2Compressor:
    """Thin adapter around the official LLMLingua-2 PromptCompressor API."""

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        force_reserve_digit: bool = False,
    ) -> None:
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:
            raise RuntimeError(
                "LLMLingua-2 baseline에는 llmlingua 패키지가 필요합니다."
            ) from exc
        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
        )
        self.force_reserve_digit = force_reserve_digit

    def compress(self, text: str, retention_rate: float) -> str:
        if not 0 < retention_rate <= 1:
            raise ValueError(f"LLMLingua-2 retention rate는 (0, 1]이어야 합니다: {retention_rate}")
        result = self.compressor.compress_prompt(
            text,
            rate=retention_rate,
            force_reserve_digit=self.force_reserve_digit,
        )
        return str(result["compressed_prompt"])


def compress_ll2_chunks(
    chunks: Mapping[str, str],
    compressor: LLMLingua2Compressor,
    retention_rate: float,
    sentence_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    selected = sentence_ids if sentence_ids is not None else list(chunks)
    rows: List[Dict[str, Any]] = []
    for sentence_id in selected:
        original = chunks[sentence_id]
        compressed = compressor.compress(original, retention_rate)
        rows.append(
            {
                "sentence_id": sentence_id,
                "method": "LLMLingua-2",
                "original": original,
                "compressed": compressed,
                "retention_rate": retention_rate,
                "n_words_original": len(original.split()),
                "n_words_compressed": len(compressed.split()),
                "n_words_dropped": len(original.split()) - len(compressed.split()),
            }
        )
    return rows


class TokenCounter:
    def __init__(self, tokenizer_name: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen CR 계산에는 transformers가 필요합니다.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def _items(text: str, pattern: re.Pattern[str]) -> List[str]:
    return pattern.findall(text)


def _preservation(
    rows: Sequence[Mapping[str, Any]], extractor
) -> Dict[str, float]:
    total = 0
    retained = 0
    supported = 0
    for row in rows:
        original = str(row["original"])
        compressed = str(row["compressed"])
        items = extractor(original)
        if items:
            supported += 1
        total += len(items)
        retained += sum(1 for item in items if _contains_item(compressed, item))
    return {
        "retained": float(retained),
        "total": float(total),
        "support": float(supported),
        "rate": retained / total if total else float("nan"),
    }


def _negation_items(text: str) -> List[str]:
    return [
        token
        for token in text.split()
        if any(cue in token for cue in NEGATION_CUES)
    ]


def _contains_item(text: str, item: str) -> bool:
    """Match an extracted item as a token-like span, not a substring."""
    item = item.strip()
    if not item:
        return False
    return re.search(
        rf"(?<!\w){re.escape(item)}(?!\w)",
        text,
        flags=re.UNICODE,
    ) is not None


def intrinsic_summary(
    rows: Sequence[Mapping[str, Any]],
    token_counter: TokenCounter,
    semantic_model: Any | None = None,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("intrinsic 평가 대상이 비어 있습니다.")
    original_tokens = sum(token_counter.count(str(row["original"])) for row in rows)
    compressed_tokens = sum(token_counter.count(str(row["compressed"])) for row in rows)
    empty_count = sum(1 for row in rows if not str(row["compressed"]).strip())
    number = _preservation(rows, lambda text: _items(text, NUMBER_RE))
    date = _preservation(rows, lambda text: _items(text, DATE_RE))
    negation = _preservation(rows, _negation_items)
    summary: Dict[str, Any] = {
        "method": rows[0].get("method", ""),
        "setting": rows[0].get("setting", ""),
        "n_samples": len(rows),
        "qwen_token_compression_ratio": 1.0 - compressed_tokens / original_tokens,
        "eojeol_compression_ratio": 1.0
        - sum(len(str(row["compressed"]).split()) for row in rows)
        / max(sum(len(str(row["original"]).split()) for row in rows), 1),
        "empty_output_count": empty_count,
        "empty_output_rate": empty_count / len(rows),
        "number_retention": number["rate"],
        "date_retention": date["rate"],
        "negation_retention": negation["rate"],
    }
    if semantic_model is not None:
        originals = [str(row["original"]) for row in rows]
        compressed = [str(row["compressed"]) for row in rows]
        original_vectors = semantic_model.encode(originals, normalize_embeddings=True)
        compressed_vectors = semantic_model.encode(compressed, normalize_embeddings=True)
        similarities = (original_vectors * compressed_vectors).sum(axis=1).tolist()
        ordered = sorted(float(value) for value in similarities)
        summary["semantic_mean"] = sum(ordered) / len(ordered)
        summary["semantic_p10"] = ordered[max(0, math.ceil(len(ordered) * 0.10) - 1)]
        summary["semantic_min"] = ordered[0]
    else:
        summary["semantic_mean"] = ""
        summary["semantic_p10"] = ""
        summary["semantic_min"] = ""
    return summary


def closest_rows(
    summaries: Sequence[Mapping[str, Any]], target_deletion_rates: Sequence[float]
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for target in target_deletion_rates:
        selected = min(
            summaries,
            key=lambda row: (
                abs(float(row["qwen_token_compression_ratio"]) - target),
                -float(row.get("semantic_mean") or float("-inf")),
            ),
        )
        result.append(
            {
                "target_deletion_rate": target,
                "selected_setting": selected["setting"],
                "actual_deletion_rate": selected["qwen_token_compression_ratio"],
                "deletion_rate_gap": abs(
                    float(selected["qwen_token_compression_ratio"]) - target
                ),
                **{
                    key: selected.get(key, "")
                    for key in (
                        "method",
                        "number_retention",
                        "date_retention",
                        "negation_retention",
                        "semantic_mean",
                        "semantic_p10",
                        "empty_output_rate",
                    )
                },
            }
        )
    return result
