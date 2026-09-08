"""Write DROP probabilities for every span in a prepared split."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from span_encoder_data import IndexedJsonlDataset, SpanBatchCollator, load_chunk_words, move_batch_to_device
from span_encoder_model import ContextualSpanClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", use_fast=True)
    dataset = IndexedJsonlDataset(args.data)
    collator = SpanBatchCollator(tokenizer, load_chunk_words(args.chunks), args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    model = ContextualSpanClassifier.from_checkpoint(args.checkpoint, "cpu").to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["span_id", "gold_label", "drop_probability", "predicted_label"],
        )
        writer.writeheader()
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch_to_device(raw_batch, device)
                output = model(**batch)
                probabilities = output["drop_probabilities"].detach().cpu().tolist()
                gold = raw_batch["labels"].tolist()
                for span_id, gold_id, probability in zip(raw_batch["span_ids"], gold, probabilities):
                    writer.writerow(
                        {
                            "span_id": span_id,
                            "gold_label": "DROP" if gold_id == 1 else "KEEP",
                            "drop_probability": f"{probability:.8f}",
                            "predicted_label": "DROP" if probability >= args.threshold else "KEEP",
                        }
                    )


if __name__ == "__main__":
    main()
