"""Shared data, training, and inference helpers for the token baseline.

The token baseline deliberately uses the same Korean backbone as the supplied
Span encoder (``klue/roberta-base`` by default).  Its only conceptual change
is the prediction unit: every eojeol is assigned a KEEP/DROP score instead of
every dependency span.

The label file is expected to contain eojeol-level PRESERVE/DISCARD labels.
The loader accepts both one-row-per-eojeol CSV files and rows containing a
list of eojeol indices plus a parallel list of labels, so it can consume the
label export used by the encoder preparation pipeline without rewriting it.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


DEFAULT_TOKEN_MODEL = "klue/roberta-base"
KEEP_ID = 0
DROP_ID = 1

_KEEP_LABELS = {"KEEP", "PRESERVE", "RETAIN", "0", "FALSE", "NO"}
_DROP_LABELS = {"DROP", "DISCARD", "REMOVE", "1", "TRUE", "YES"}
_PUNCT_ONLY = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def _open_text(path: Path, mode: str = "rt"):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8-sig", newline="")


def _pick(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _parse_value(value: Any) -> List[Any]:
    """Parse a scalar/list CSV value without assuming one serialization."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    if parsed is not None and not isinstance(parsed, (dict, set)):
        return [parsed]
    stripped = text.strip("[]()")
    if not stripped:
        return []
    if "," in stripped:
        return [item.strip().strip("'\"") for item in stripped.split(",") if item.strip()]
    return [item for item in stripped.split() if item]


def normalize_binary_label(value: Any) -> int:
    text = str(value).strip().upper()
    if text in _KEEP_LABELS:
        return KEEP_ID
    if text in _DROP_LABELS:
        return DROP_ID
    raise ValueError(
        f"알 수 없는 token label입니다: {value!r}. "
        "PRESERVE/DISCARD 또는 KEEP/DROP을 사용하세요."
    )


def load_eojeol_labels(path: Path) -> Dict[str, Dict[int, int]]:
    """Load ``sentence_id -> eojeol_index -> {KEEP=0,DROP=1}`` labels."""
    index_names = (
        "eojeol_index",
        "word_index",
        "token_index",
        "index",
        "position",
        "eojeol_indices",
        "word_ids",
    )
    label_names = (
        "eojeol_labels",
        "labels",
        "eojeol_label",
        "word_label",
        "token_label",
        "label",
        "auto_label",
    )
    result: Dict[str, Dict[int, int]] = {}
    with _open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"label CSV header를 읽지 못했습니다: {path}")
        for row_number, row in enumerate(reader, start=2):
            sentence_id = _pick(
                row,
                ("sentence_id", "chunk_id", "passage_id", "chunk", "id"),
            )
            if sentence_id is None:
                raise ValueError(f"sentence_id 열이 없습니다: {path}:{row_number}")
            sentence_id = str(sentence_id)
            raw_labels = _pick(row, label_names)
            if raw_labels is None:
                raise ValueError(f"label 열이 없습니다: {path}:{row_number}")
            labels = _parse_value(raw_labels)
            raw_indices = _pick(row, index_names)
            indices = _parse_value(raw_indices) if raw_indices is not None else []

            # A plural/list label column represents a sequence.  If indices
            # are omitted, the sequence is assumed to start at zero.
            if len(labels) > 1 or len(indices) > 1:
                if not indices:
                    indices = list(range(len(labels)))
                if len(indices) != len(labels):
                    raise ValueError(
                        f"indices와 labels 길이가 다릅니다: {path}:{row_number} "
                        f"({len(indices)} != {len(labels)})"
                    )
            else:
                if not indices:
                    raise ValueError(
                        f"eojeol index가 없습니다: {path}:{row_number}"
                    )
                indices = [indices[0]]
                labels = [labels[0]]

            sentence_labels = result.setdefault(sentence_id, {})
            for raw_index, raw_label in zip(indices, labels):
                try:
                    index = int(raw_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"eojeol index가 정수가 아닙니다: {path}:{row_number}: {raw_index!r}"
                    ) from exc
                label = normalize_binary_label(raw_label)
                previous = sentence_labels.get(index)
                if previous is not None and previous != label:
                    raise ValueError(
                        f"같은 eojeol에 서로 다른 label이 있습니다: "
                        f"{sentence_id}[{index}]"
                    )
                sentence_labels[index] = label
    if not result:
        raise ValueError(f"읽은 eojeol label이 없습니다: {path}")
    return result


@dataclass(frozen=True)
class TokenExample:
    sentence_id: str
    document_id: str
    words: Tuple[str, ...]
    labels: Tuple[int, ...]
    split: str


def load_split_manifest(path: Path) -> Dict[str, str]:
    """Load document-level train/validation/test assignments."""
    result: Dict[str, str] = {}
    with _open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            document_id = _pick(row, ("document_id", "doc_id", "document"))
            split = _pick(row, ("split", "partition", "set"))
            if document_id is None or split is None:
                raise ValueError(f"split manifest 열이 없습니다: {path}:{row_number}")
            normalized = str(split).lower()
            if normalized == "val":
                normalized = "validation"
            if normalized not in {"train", "validation", "test"}:
                raise ValueError(f"알 수 없는 split입니다: {split!r}")
            result[str(document_id)] = normalized
    if not result:
        raise ValueError(f"split manifest가 비어 있습니다: {path}")
    return result


def _deterministic_split(document_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"


def load_token_examples(
    chunks_path: Path,
    labels_path: Path,
    split_manifest: Path | None = None,
    seed: int = 42,
) -> List[TokenExample]:
    """Join chunks and eojeol labels into document-level split examples."""
    labels = load_eojeol_labels(labels_path)
    split_by_document = (
        load_split_manifest(split_manifest) if split_manifest is not None else {}
    )
    examples: List[TokenExample] = []
    with _open_text(chunks_path, "rt") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            sentence_id = _pick(row, ("sentence_id", "chunk_id", "passage_id", "id"))
            text = _pick(row, ("sentence", "context", "text", "chunk"))
            if sentence_id is None or text is None:
                raise ValueError(f"chunks 열이 없습니다: {chunks_path}:{row_number}")
            sentence_id = str(sentence_id)
            document_id = str(
                _pick(row, ("document_id", "doc_id", "document"))
                or sentence_id.split("__", 1)[0]
            )
            words = tuple(str(text).split())
            if sentence_id not in labels:
                raise ValueError(f"label이 없는 chunk입니다: {sentence_id}")
            by_index = labels[sentence_id]
            missing = [index for index in range(len(words)) if index not in by_index]
            if missing:
                raise ValueError(
                    f"chunk의 eojeol label이 누락되었습니다: {sentence_id}; "
                    f"첫 누락 index={missing[0]}"
                )
            extra = [index for index in by_index if index < 0 or index >= len(words)]
            if extra:
                raise ValueError(
                    f"chunk 길이를 벗어난 eojeol label입니다: {sentence_id}; "
                    f"첫 초과 index={extra[0]}"
                )
            split = split_by_document.get(document_id) or _deterministic_split(
                document_id, seed
            )
            examples.append(
                TokenExample(
                    sentence_id=sentence_id,
                    document_id=document_id,
                    words=words,
                    labels=tuple(by_index[index] for index in range(len(words))),
                    split=split,
                )
            )
    if not examples:
        raise ValueError(f"읽은 chunk가 없습니다: {chunks_path}")
    return examples


class TokenClassificationDataset:
    """Lazy fast-tokenizer dataset with eojeol labels copied to subwords."""

    def __init__(self, examples: Sequence[TokenExample], tokenizer: Any, max_length: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        encoding = self.tokenizer(
            list(example.words),
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids()
        encoding["labels"] = [
            -100 if word_id is None or word_id >= len(example.labels) else example.labels[word_id]
            for word_id in word_ids
        ]
        return encoding


def _drop_label_id(model: Any) -> int:
    id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
    for raw_id, label in id2label.items():
        if str(label).upper() in _DROP_LABELS:
            return int(raw_id)
    return DROP_ID


class TokenBaselineCompressor:
    """Score and compress whole eojeols with a trained token classifier."""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer_name: str | None = None,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "token baseline에는 torch와 transformers가 필요합니다."
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda를 지정했지만 CUDA를 사용할 수 없습니다.")
        self.device = torch.device(device)
        tokenizer_source = tokenizer_name or str(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("token baseline은 fast tokenizer가 필요합니다.")
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.drop_id = _drop_label_id(self.model)
        self.max_length = max_length

    def score_words(self, words: Sequence[str]) -> Tuple[List[float], List[int]]:
        import torch

        encoding = self.tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        model_inputs = {key: value.to(self.device) for key, value in encoding.items()}
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)[:, self.drop_id].cpu().tolist()
        sums = [0.0] * len(words)
        counts = [0] * len(words)
        for word_id, probability in zip(word_ids, probabilities):
            if word_id is not None and 0 <= word_id < len(words):
                sums[word_id] += float(probability)
                counts[word_id] += 1
        scores = [sums[index] / counts[index] if counts[index] else 0.0 for index in range(len(words))]
        truncated = [index for index, count in enumerate(counts) if count == 0]
        return scores, truncated

    def compress(self, text: str, threshold: float) -> str:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"token threshold는 0~1이어야 합니다: {threshold}")
        words = str(text).split()
        scores, truncated = self.score_words(words)
        truncated_set = set(truncated)
        kept = [
            word
            for index, (word, score) in enumerate(zip(words, scores))
            if index in truncated_set or score < threshold or _PUNCT_ONLY.fullmatch(word)
        ]
        return " ".join(kept)

    def score_text(self, text: str) -> Dict[str, Any]:
        words = str(text).split()
        scores, truncated = self.score_words(words)
        return {
            "words": words,
            "drop_probabilities": scores,
            "truncated_word_indices": truncated,
        }
