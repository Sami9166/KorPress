"""Indexed JSONL dataset and span-aware contextual collation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


LABEL_TO_ID = {"KEEP": 0, "DROP": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def load_chunk_words(chunks_csv: str | Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with Path(chunks_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["sentence_id"]] = row["sentence"].split()
    return result


class IndexedJsonlDataset(Dataset):
    """Random-access JSONL without loading one million spans into RAM."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offsets: list[int] = []
        self._handle = None
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline().decode("utf-8"))


class SpanBatchCollator:
    """Build target-span windows with context-only surrounding utterances.

    The surrounding words are encoded once per target window and influence the
    target span representation, but they never receive a span label or loss.
    """

    def __init__(self, tokenizer, chunk_words: dict[str, list[str]], max_length: int = 512):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for word_ids() alignment")
        self.tokenizer = tokenizer
        self.chunk_words = chunk_words
        self.max_length = max_length
        self._piece_count_cache: dict[str, list[int]] = {}

    def _piece_counts(self, sentence_id: str) -> list[int]:
        cached = self._piece_count_cache.get(sentence_id)
        if cached is not None:
            return cached
        words = self.chunk_words[sentence_id]
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=False,
        )
        counts = [0] * len(words)
        for word_id in encoded.word_ids():
            if word_id is not None:
                counts[word_id] += 1
        if any(count == 0 for count in counts):
            raise ValueError(f"Tokenizer produced an empty word in {sentence_id}")
        self._piece_count_cache[sentence_id] = counts
        return counts

    @staticmethod
    def _split_target_ranges(
        counts: list[int], start: int, end: int, budget: int
    ) -> list[tuple[int, int]]:
        """Split an overlong target utterance at word boundaries."""
        ranges: list[tuple[int, int]] = []
        cursor = start
        while cursor < end:
            next_cursor = cursor
            used = 0
            while next_cursor < end and used + counts[next_cursor] <= budget:
                used += counts[next_cursor]
                next_cursor += 1
            if next_cursor == cursor:
                raise ValueError(
                    "하나의 어절이 subword budget보다 길어 window로 나눌 수 없습니다: "
                    f"word_index={cursor}, budget={budget}"
                )
            ranges.append((cursor, next_cursor))
            cursor = next_cursor
        return ranges

    @staticmethod
    def _context_window(
        counts: list[int], target_start: int, target_end: int, budget: int
    ) -> tuple[int, int]:
        """Add neighboring words while leaving the target range untouched."""
        used = sum(counts[target_start:target_end])
        left, right = target_start - 1, target_end
        take_left = True
        while left >= 0 or right < len(counts):
            candidates = (left, right) if take_left else (right, left)
            added = False
            for position in candidates:
                if position < 0 or position >= len(counts):
                    continue
                if used + counts[position] <= budget:
                    used += counts[position]
                    if position == left:
                        left -= 1
                    else:
                        right += 1
                    added = True
                    take_left = not take_left
                    break
            if not added:
                break
        return left + 1, right

    def _windows_for_record(
        self, record: dict[str, Any]
    ) -> list[tuple[int, int, dict[str, Any]]]:
        sentence_id = record["sentence_id"]
        counts = self._piece_counts(sentence_id)
        start = int(record["utterance_start"])
        end = int(record["utterance_end"])
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if not 0 <= start < end <= len(counts):
            raise ValueError(f"잘못된 utterance 범위입니다: {(start, end)}")
        if budget <= 0:
            raise ValueError(f"max_length가 special token보다 작습니다: {self.max_length}")

        target_ranges = self._split_target_ranges(counts, start, end, budget)
        remaining = list(record["spans"])
        windows: list[tuple[int, int, dict[str, Any]]] = []
        for target_start, target_end in target_ranges:
            selected = [
                span
                for span in remaining
                if all(
                    target_start <= int(index) < target_end
                    for index in span["eojeol_indices"]
                )
            ]
            if not selected:
                continue
            selected_ids = {str(span["span_id"]) for span in selected}
            remaining = [
                span for span in remaining if str(span["span_id"]) not in selected_ids
            ]
            begin, finish = self._context_window(
                counts, target_start, target_end, budget
            )
            expanded = dict(record)
            expanded["spans"] = selected
            windows.append((begin, finish, expanded))

        # A span crossing a split boundary gets a dedicated target window. This
        # keeps every span intact and emits each span_id exactly once.
        for span in remaining:
            indices = [int(index) for index in span["eojeol_indices"]]
            span_start, span_end = min(indices), max(indices) + 1
            span_subwords = sum(counts[span_start:span_end])
            if span_subwords > budget:
                raise ValueError(
                    f"span이 subword budget보다 깁니다: {span['span_id']} "
                    f"({span_subwords} > {budget})"
                )
            begin, finish = self._context_window(
                counts, span_start, span_end, budget
            )
            expanded = dict(record)
            expanded["spans"] = [span]
            windows.append((begin, finish, expanded))
        return windows

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        windows: list[list[str]] = []
        bounds: list[tuple[int, int]] = []
        expanded_records: list[dict[str, Any]] = []
        for record in records:
            for begin, end, expanded in self._windows_for_record(record):
                bounds.append((begin, end))
                windows.append(self.chunk_words[record["sentence_id"]][begin:end])
                expanded_records.append(expanded)

        encoded = self.tokenizer(
            windows,
            is_split_into_words=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        all_masks: list[torch.Tensor] = []
        labels: list[int] = []
        span_ids: list[str] = []
        for batch_index, (record, (begin, _)) in enumerate(
            zip(expanded_records, bounds)
        ):
            word_ids = encoded.word_ids(batch_index=batch_index)
            masks = torch.zeros((len(record["spans"]), len(word_ids)), dtype=torch.bool)
            for span_index, span in enumerate(record["spans"]):
                local_positions = {int(i) - begin for i in span["eojeol_indices"]}
                for token_index, word_id in enumerate(word_ids):
                    if word_id is not None and word_id in local_positions:
                        masks[span_index, token_index] = True
                if not masks[span_index].any():
                    raise ValueError(f"Empty aligned span: {span['span_id']}")
                labels.append(LABEL_TO_ID[span["label"]])
                span_ids.append(span["span_id"])
            all_masks.append(masks)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "span_token_masks": all_masks,
            "labels": torch.tensor(labels, dtype=torch.long),
            "span_ids": span_ids,
            "records": expanded_records,
            "window_bounds": bounds,
        }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "span_token_masks": [mask.to(device) for mask in batch["span_token_masks"]],
        "labels": batch["labels"].to(device),
    }
