"""Run one real span batch through tokenizer, pooling, and classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from transformers import AutoTokenizer

from span_encoder_data import IndexedJsonlDataset, SpanBatchCollator, load_chunk_words, move_batch_to_device
from span_encoder_model import ContextualSpanClassifier, SpanEncoderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--model-name", default="klue/roberta-base")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    dataset = IndexedJsonlDataset(args.data_dir / "train.jsonl")
    record = dataset[0]
    collator = SpanBatchCollator(
        tokenizer, load_chunk_words(args.chunks), max_length=args.max_length
    )
    raw_batch = collator([record])
    batch = move_batch_to_device(raw_batch, device)
    model = ContextualSpanClassifier(
        SpanEncoderConfig(model_name=args.model_name, pooling="mean_max")
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(**batch)

    probabilities = output["drop_probabilities"].cpu()
    report = {
        "status": "OK",
        "device": str(device),
        "utterance_uid": record["utterance_uid"],
        "context_subwords": int(raw_batch["attention_mask"].sum()),
        "span_count": len(record["spans"]),
        "span_vector_shape": list(output["span_vectors"].shape),
        "logit_shape": list(output["logits"].shape),
        "drop_probability_min": float(probabilities.min()),
        "drop_probability_max": float(probabilities.max()),
        "loss": float(output["loss"].cpu()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
