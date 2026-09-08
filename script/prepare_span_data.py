"""Convert projected span CSV into document-split utterance JSONL files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def load_chunks(path: Path) -> dict[str, dict]:
    chunks: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            lines = [line.strip() for line in row["sentence"].splitlines() if line.strip()]
            if len(lines) != int(row["utterance_count"]):
                raise ValueError(
                    f"Utterance count mismatch in {row['sentence_id']}: "
                    f"CSV={row['utterance_count']}, lines={len(lines)}"
                )
            boundaries = []
            cursor = 0
            for line in lines:
                word_count = len(line.split())
                boundaries.append((cursor, cursor + word_count))
                cursor += word_count
            chunks[row["sentence_id"]] = {
                "document_id": row["document_id"],
                "field": row["field"],
                "major": row["major"],
                "boundaries": boundaries,
                "word_count": cursor,
                "words": row["sentence"].split(),
            }
    return chunks


def make_document_split(
    chunks: dict[str, dict], seed: int
) -> tuple[dict[str, str], list[dict[str, str]]]:
    doc_field: dict[str, str] = {}
    for meta in chunks.values():
        previous = doc_field.get(meta["document_id"])
        if previous is not None and previous != meta["field"]:
            raise ValueError(
                f"Document {meta['document_id']} has multiple fields: "
                f"{previous}, {meta['field']}"
            )
        doc_field[meta["document_id"]] = meta["field"]
    by_field: dict[str, list[str]] = defaultdict(list)
    for document_id, field in doc_field.items():
        by_field[field].append(document_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for field, document_ids in sorted(by_field.items()):
        rng.shuffle(document_ids)
        n = len(document_ids)
        n_valid = max(1, round(n * 0.1)) if n >= 3 else 0
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        while n_valid + n_test >= n:
            if n_test >= n_valid and n_test:
                n_test -= 1
            elif n_valid:
                n_valid -= 1
        for index, document_id in enumerate(document_ids):
            split = (
                "validation"
                if index < n_valid
                else "test"
                if index < n_valid + n_test
                else "train"
            )
            assignment[document_id] = split
            rows.append({"document_id": document_id, "field": field, "split": split})
    return assignment, rows


def grouped_rows(reader: Iterator[dict]) -> Iterator[list[dict]]:
    current_key = None
    group: list[dict] = []
    finished: set[tuple[str, str]] = set()
    for row in reader:
        key = (row["sentence_id"], row["utterance_uid"])
        if current_key is not None and key != current_key:
            finished.add(current_key)
            yield group
            group = []
            if key in finished:
                raise ValueError(f"Span rows are not grouped by utterance: {key}")
        current_key = key
        group.append(row)
    if group:
        yield group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-fully-dropped", action="store_true")
    parser.add_argument("--limit-utterances", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(args.chunks)
    split_by_doc, split_rows = make_document_split(chunks, args.seed)
    split_manifest_tmp = args.output_dir / "split_manifest.csv.tmp"
    with split_manifest_tmp.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["document_id", "field", "split"])
        writer.writeheader()
        writer.writerows(split_rows)

    temporary_paths = {
        split: args.output_dir / f"{split}.jsonl.tmp"
        for split in ("train", "validation", "test")
    }
    outputs = {split: path.open("w", encoding="utf-8") for split, path in temporary_paths.items()}
    utterance_counts = Counter()
    span_counts = Counter()
    label_counts = Counter()
    skipped_fully_dropped = 0
    seen = 0
    try:
        with open_text(args.spans) as handle:
            for group in grouped_rows(csv.DictReader(handle)):
                if args.limit_utterances and seen >= args.limit_utterances:
                    break
                first = group[0]
                fully_dropped = first["utterance_fully_dropped"].lower() == "true"
                if args.exclude_fully_dropped and fully_dropped:
                    skipped_fully_dropped += 1
                    continue
                sentence_id = first["sentence_id"]
                meta = chunks[sentence_id]
                utterance_index = int(first["utterance_index_in_chunk"])
                try:
                    utterance_start, utterance_end = meta["boundaries"][utterance_index]
                except IndexError as exc:
                    raise ValueError(
                        f"Invalid utterance index {utterance_index} for {sentence_id}"
                    ) from exc

                spans = []
                for row in group:
                    indices = [int(i) for i in json.loads(row["eojeol_indices"])]
                    if (
                        not indices
                        or min(indices) < utterance_start
                        or max(indices) >= utterance_end
                    ):
                        raise ValueError(f"Out-of-utterance span: {row['span_id']}")
                    if len(indices) != int(row["size"]) or len(indices) != len(set(indices)):
                        raise ValueError(f"Invalid span size/indices: {row['span_id']}")
                    surface = " ".join(meta["words"][i] for i in indices)
                    if surface != row["span"]:
                        raise ValueError(
                            f"Span surface mismatch: {row['span_id']} "
                            f"expected={surface!r}, csv={row['span']!r}"
                        )
                    if row["auto_label"] not in {"KEEP", "DROP"}:
                        raise ValueError(f"Invalid label: {row['span_id']}")
                    spans.append(
                        {
                            "span_id": row["span_id"],
                            "text": row["span"],
                            "eojeol_indices": indices,
                            "size": int(row["size"]),
                            "is_contiguous": row["is_contiguous"].lower() == "true",
                            "label": row["auto_label"],
                        }
                    )

                document_id = meta["document_id"]
                split = split_by_doc[document_id]
                record = {
                    "sentence_id": sentence_id,
                    "document_id": document_id,
                    "field": meta["field"],
                    "major": meta["major"],
                    "utterance_uid": first["utterance_uid"],
                    "utterance_index_in_chunk": utterance_index,
                    "utterance_start": utterance_start,
                    "utterance_end": utterance_end,
                    "utterance_fully_dropped": fully_dropped,
                    "spans": spans,
                }
                outputs[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                utterance_counts[split] += 1
                span_counts[split] += len(spans)
                for span in spans:
                    label_counts[f"{split}:{span['label']}"] += 1
                seen += 1
    finally:
        for handle in outputs.values():
            handle.close()

    summary = {
        "source_chunks": str(args.chunks.resolve()),
        "source_spans": str(args.spans.resolve()),
        "seed": args.seed,
        "exclude_fully_dropped": args.exclude_fully_dropped,
        "skipped_fully_dropped_utterances": skipped_fully_dropped,
        "utterances": dict(utterance_counts),
        "spans": dict(span_counts),
        "labels": dict(label_counts),
    }
    summary_tmp = args.output_dir / "summary.json.tmp"
    summary_tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for split, temporary_path in temporary_paths.items():
        temporary_path.replace(args.output_dir / f"{split}.jsonl")
    split_manifest_tmp.replace(args.output_dir / "split_manifest.csv")
    summary_tmp.replace(args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
