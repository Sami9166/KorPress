"""Verify that prepared JSONL counts match summary.json."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.data_dir / "summary.json").read_text(encoding="utf-8"))

    manifest = list(
        csv.DictReader((args.data_dir / "split_manifest.csv").open(encoding="utf-8-sig"))
    )
    documents = Counter(row["split"] for row in manifest)
    observed_utterances = Counter()
    observed_spans = Counter()
    observed_labels = Counter()
    for split in ("train", "validation", "test"):
        with (args.data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record["spans"]:
                    raise ValueError(f"Empty span group: {record['utterance_uid']}")
                observed_utterances[split] += 1
                observed_spans[split] += len(record["spans"])
                for span in record["spans"]:
                    observed_labels[f"{split}:{span['label']}"] += 1

    expected = {
        "utterances": Counter(summary["utterances"]),
        "spans": Counter(summary["spans"]),
        "labels": Counter(summary["labels"]),
    }
    observed = {
        "utterances": observed_utterances,
        "spans": observed_spans,
        "labels": observed_labels,
    }
    if observed != expected:
        raise ValueError(f"Prepared data mismatch\nexpected={expected}\nobserved={observed}")
    print(
        json.dumps(
            {
                "status": "OK",
                "documents": dict(documents),
                "utterances": dict(observed_utterances),
                "spans": dict(observed_spans),
                "labels": dict(observed_labels),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
