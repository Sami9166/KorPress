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
    """Build a max-length context window that always contains the target utterance."""

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

    def _window(self, record: dict[str, Any]) -> tuple[int, int]:
        sentence_id = record["sentence_id"]
        counts = self._piece_counts(sentence_id)
        start = int(record["utterance_start"])
        end = int(record["utterance_end"])
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        used = sum(counts[start:end])
        if used > budget:
            raise ValueError(
                f"Target utterance exceeds {budget} subwords: "
                f"{record['utterance_uid']} ({used})"
            )

        left, right = start - 1, end
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

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        windows: list[list[str]] = []
        bounds: list[tuple[int, int]] = []
        for record in records:
            begin, end = self._window(record)
            bounds.append((begin, end))
            windows.append(self.chunk_words[record["sentence_id"]][begin:end])

        encoded = self.tokenizer(
            windows,
            is_split_into_words=True,
            padding=True,
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        all_masks: list[torch.Tensor] = []
        labels: list[int] = []
        span_ids: list[str] = []
        for batch_index, (record, (begin, _)) in enumerate(zip(records, bounds)):
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
            "records": records,
            "window_bounds": bounds,
        }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "span_token_masks": [mask.to(device) for mask in batch["span_token_masks"]],
        "labels": batch["labels"].to(device),
    }
