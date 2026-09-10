"""Subword token baseline data, training, and inference helpers.

The baseline uses the same Korean backbone as the Span encoder, but predicts
KEEP/DROP for each tokenizer subword.  The loader still accepts the old
eojeol-label CSV for backward compatibility; ``subword_labels.csv.gz`` is the
preferred training input.
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


def _is_punctuation_piece(piece: str) -> bool:
    piece = re.sub(r"^(?:##|Ġ|▁)+", "", str(piece)).strip()
    return bool(piece) and bool(_PUNCT_ONLY.fullmatch(piece))


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


def load_indexed_labels(path: Path) -> Dict[str, Dict[int, int]]:
    """Load ``sentence_id -> index -> {KEEP=0,DROP=1}`` labels."""
    index_names = (
        "subword_index",
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
                        f"label index가 없습니다: {path}:{row_number}"
                    )
                indices = [indices[0]]
                labels = [labels[0]]

            sentence_labels = result.setdefault(sentence_id, {})
            for raw_index, raw_label in zip(indices, labels):
                try:
                    index = int(raw_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"label index가 정수가 아닙니다: {path}:{row_number}: {raw_index!r}"
                    ) from exc
                label = normalize_binary_label(raw_label)
                previous = sentence_labels.get(index)
                if previous is not None and previous != label:
                    raise ValueError(
                        f"같은 index에 서로 다른 label이 있습니다: "
                        f"{sentence_id}[{index}]"
                    )
                sentence_labels[index] = label
    if not result:
        raise ValueError(f"읽은 label이 없습니다: {path}")
    return result


def load_eojeol_labels(path: Path) -> Dict[str, Dict[int, int]]:
    """Backward-compatible name for the indexed-label loader."""
    return load_indexed_labels(path)


def detect_label_unit(path: Path, requested: str = "auto") -> str:
    """Return ``word`` for legacy labels or ``subword`` for teacher labels."""
    if requested not in {"auto", "word", "subword"}:
        raise ValueError(f"알 수 없는 label unit입니다: {requested}")
    if requested != "auto":
        return requested
    with _open_text(path, "rt") as handle:
        fieldnames = next(csv.reader(handle), [])
    names = {str(name).strip() for name in fieldnames}
    return "subword" if {"token_index", "subword_index"} & names else "word"


def text_words_and_utterance_bounds(
    text: str,
) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, int], ...]]:
    """Split a chunk like the Span pipeline: one non-empty line per utterance."""
    words: List[str] = []
    bounds: List[Tuple[int, int]] = []
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        lines = [str(text).strip()]
    for line in lines:
        line_words = line.split()
        if not line_words:
            continue
        start = len(words)
        words.extend(line_words)
        bounds.append((start, len(words)))
    return tuple(words), tuple(bounds or [(0, 0)])


def _piece_counts(tokenizer: Any, words: Sequence[str]) -> List[int]:
    encoded = tokenizer(
        list(words),
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
    )
    counts = [0] * len(words)
    for word_id in encoded.word_ids():
        if word_id is not None:
            counts[word_id] += 1
    if any(count == 0 for count in counts):
        raise ValueError("tokenizer가 subword를 생성하지 않은 어절이 있습니다.")
    return counts


def _utterance_windows(
    tokenizer: Any,
    words: Sequence[str],
    utterance_bounds: Sequence[Tuple[int, int]],
    max_length: int,
) -> List[Tuple[int, int, int, int]]:
    """Build the same target-utterance-centered windows as SpanBatchCollator."""
    counts = _piece_counts(tokenizer, words)
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= 0:
        raise ValueError(f"max_length가 special token보다 작습니다: {max_length}")
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)

    windows: List[Tuple[int, int, int, int]] = []
    for start, end in utterance_bounds:
        if not 0 <= start < end <= len(words):
            raise ValueError(f"잘못된 utterance 범위입니다: {(start, end)}")
        used = offsets[end] - offsets[start]
        if used > budget:
            raise ValueError(
                f"Target utterance exceeds {budget} subwords: {(start, end)} ({used})"
            )
        left, right = start - 1, end
        take_left = True
        while left >= 0 or right < len(words):
            candidates = (left, right) if take_left else (right, left)
            added = False
            for position in candidates:
                if position < 0 or position >= len(words):
                    continue
                candidate_start = offsets[position]
                candidate_end = offsets[position + 1]
                if used + candidate_end - candidate_start <= budget:
                    used += candidate_end - candidate_start
                    if position == left:
                        left -= 1
                    else:
                        right += 1
                    added = True
                    take_left = not take_left
                    break
            if not added:
                break
        begin, finish = left + 1, right
        windows.append((begin, finish, offsets[begin], offsets[finish]))
    return windows


@dataclass(frozen=True)
class TokenExample:
    sentence_id: str
    document_id: str
    words: Tuple[str, ...]
    labels: Tuple[int, ...]
    split: str
    subword_labels: Tuple[int, ...] | None = None
    utterance_bounds: Tuple[Tuple[int, int], ...] = ()


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
    label_unit: str = "auto",
) -> List[TokenExample]:
    """Join chunks and word- or subword-level labels into split examples."""
    label_unit = detect_label_unit(labels_path, label_unit)
    labels = load_indexed_labels(labels_path)
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
            words, utterance_bounds = text_words_and_utterance_bounds(str(text))
            if sentence_id not in labels:
                raise ValueError(f"label이 없는 chunk입니다: {sentence_id}")
            by_index = labels[sentence_id]
            if label_unit == "subword":
                if min(by_index) < 0:
                    raise ValueError(f"음수 subword index가 있습니다: {sentence_id}")
                first_missing = [index for index in range(max(by_index) + 1) if index not in by_index]
                if first_missing:
                    raise ValueError(
                        f"chunk의 subword label이 누락되었습니다: {sentence_id}; "
                        f"첫 누락 index={first_missing[0]}"
                    )
                subword_labels = tuple(by_index[index] for index in range(max(by_index) + 1))
                word_labels: Tuple[int, ...] = ()
            else:
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
                word_labels = tuple(by_index[index] for index in range(len(words)))
                subword_labels = None
            split = split_by_document.get(document_id) or _deterministic_split(
                document_id, seed
            )
            examples.append(
                TokenExample(
                    sentence_id=sentence_id,
                    document_id=document_id,
                    words=words,
                    labels=word_labels,
                    split=split,
                    subword_labels=subword_labels,
                    utterance_bounds=utterance_bounds,
                )
            )
    if not examples:
        raise ValueError(f"읽은 chunk가 없습니다: {chunks_path}")
    return examples


class TokenClassificationDataset:
    """Utterance-windowed dataset for word or precomputed subword labels."""

    def __init__(self, examples: Sequence[TokenExample], tokenizer: Any, max_length: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.windows: List[Tuple[TokenExample, int, int, int, int]] = []
        for example in self.examples:
            bounds = example.utterance_bounds or ((0, len(example.words)),)
            for begin, end, subword_begin, subword_end in _utterance_windows(
                tokenizer,
                example.words,
                bounds,
                max_length,
            ):
                self.windows.append((example, begin, end, subword_begin, subword_end))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example, begin, end, subword_begin, subword_end = self.windows[index]
        encoding = self.tokenizer(
            list(example.words[begin:end]),
            is_split_into_words=True,
            truncation=False,
        )
        word_ids = encoding.word_ids()
        if example.subword_labels is None:
            word_labels = example.labels[begin:end]
            encoding["labels"] = [
                -100
                if word_id is None or word_id >= len(word_labels)
                else word_labels[word_id]
                for word_id in word_ids
            ]
        else:
            subword_labels = example.subword_labels[subword_begin:subword_end]
            labels = []
            subword_index = 0
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                    continue
                if subword_index >= len(subword_labels):
                    raise ValueError(
                        f"tokenizer subword 수가 label보다 많습니다: {example.sentence_id}"
                    )
                labels.append(subword_labels[subword_index])
                subword_index += 1
            encoding["labels"] = labels
        return encoding


def _drop_label_id(model: Any) -> int:
    id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
    for raw_id, label in id2label.items():
        if str(label).upper() in _DROP_LABELS:
            return int(raw_id)
    return DROP_ID


class TokenBaselineCompressor:
    """Score and compress tokenizer subwords with a trained classifier."""

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

    def _score_window(self, words: Sequence[str]) -> Tuple[List[float], List[int]]:
        import torch

        encoding = self.tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        positions = [index for index, word_id in enumerate(word_ids) if word_id is not None]
        visible_ids = encoding["input_ids"][0].detach().cpu().tolist()
        visible_ids = [visible_ids[index] for index in positions]
        model_inputs = {key: value.to(self.device) for key, value in encoding.items()}
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)[:, self.drop_id].cpu().tolist()
        return [float(probabilities[index]) for index in positions], visible_ids

    def score_tokens(
        self,
        words: Sequence[str],
        utterance_bounds: Sequence[Tuple[int, int]] | None = None,
    ) -> Tuple[List[float], List[int], List[int], List[str]]:
        """Score every subword through Span-style utterance-centered windows."""
        full_encoding = self.tokenizer(
            list(words),
            is_split_into_words=True,
            add_special_tokens=False,
        )
        full_ids = full_encoding["input_ids"]
        if full_ids and isinstance(full_ids[0], list):
            full_ids = full_ids[0]
        full_ids = [int(token_id) for token_id in full_ids]
        bounds = utterance_bounds or ((0, len(words)),)
        windows = _utterance_windows(self.tokenizer, words, bounds, self.max_length)
        score_sums = [0.0] * len(full_ids)
        score_counts = [0] * len(full_ids)
        for begin, end, subword_begin, subword_end in windows:
            local_scores, local_ids = self._score_window(words[begin:end])
            expected_ids = full_ids[subword_begin:subword_end]
            if local_ids != expected_ids:
                raise ValueError("학습 tokenizer와 추론 tokenizer의 subword 정렬이 다릅니다.")
            for offset, score in enumerate(local_scores):
                global_index = subword_begin + offset
                score_sums[global_index] += score
                score_counts[global_index] += 1
        missing = [index for index, count in enumerate(score_counts) if count == 0]
        if missing:
            raise ValueError(f"window에서 점수를 얻지 못한 subword가 있습니다: {missing[0]}")
        scores = [total / count for total, count in zip(score_sums, score_counts)]
        truncated: List[int] = []
        token_texts = self.tokenizer.convert_ids_to_tokens(full_ids)
        return scores, full_ids, truncated, token_texts

    def score_words(self, words: Sequence[str]) -> Tuple[List[float], List[int]]:
        """Compatibility alias; returned indices are now subword indices."""
        scores, _token_ids, truncated, _tokens = self.score_tokens(words)
        return scores, truncated

    def compress(self, text: str, threshold: float) -> str:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"token threshold는 0~1이어야 합니다: {threshold}")
        words, utterance_bounds = text_words_and_utterance_bounds(str(text))
        scores, token_ids, truncated, token_texts = self.score_tokens(words, utterance_bounds)
        truncated_set = set(truncated)
        kept_ids = [
            token_id
            for index, (token_id, score) in enumerate(zip(token_ids, scores))
            if index in truncated_set
            or score < threshold
            or _is_punctuation_piece(token_texts[index])
        ]
        return self.tokenizer.decode(
            kept_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def score_text(self, text: str) -> Dict[str, Any]:
        words, utterance_bounds = text_words_and_utterance_bounds(str(text))
        scores, token_ids, truncated, token_texts = self.score_tokens(words, utterance_bounds)
        return {
            "tokens": token_texts,
            "token_ids": token_ids,
            "drop_probabilities": scores,
            "truncated_token_indices": truncated,
        }
