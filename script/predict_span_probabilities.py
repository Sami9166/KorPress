"""Run the supplied contextual span encoder and write DROP probabilities.

The encoder source is intentionally not copied into KorPress.  Pass the
extracted ``span_encoder`` directory from the encoder bundle with
``--encoder-root``.  This keeps the experiment directory independent while
using the exact model/data implementation that produced the existing results.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import sys
from pathlib import Path


def _load_encoder_modules(root: Path):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"encoder root를 찾을 수 없습니다: {root}")
    sys.path.insert(0, str(root))
    try:
        data_module = importlib.import_module("data")
        model_module = importlib.import_module("model")
    except ImportError as exc:
        raise ImportError(
            "encoder-root에는 data.py와 model.py가 있어야 합니다. "
            "span encoder bundle을 먼저 압축 해제하세요."
        ) from exc
    return data_module, model_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    data_module, model_module = _load_encoder_modules(args.encoder_root)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda를 지정했지만 CUDA를 사용할 수 없습니다.")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", use_fast=True)
    dataset = data_module.IndexedJsonlDataset(args.data)
    collator = data_module.SpanBatchCollator(
        tokenizer,
        data_module.load_chunk_words(args.chunks),
        args.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    model = model_module.ContextualSpanClassifier.from_checkpoint(
        args.checkpoint, map_location="cpu"
    ).to(device)
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
                batch = data_module.move_batch_to_device(raw_batch, device)
                output = model(**batch)
                probabilities = output["drop_probabilities"].detach().cpu().tolist()
                gold = raw_batch["labels"].tolist()
                for span_id, gold_id, probability in zip(
                    raw_batch["span_ids"], gold, probabilities
                ):
                    writer.writerow(
                        {
                            "span_id": span_id,
                            "gold_label": "DROP" if gold_id == 1 else "KEEP",
                            "drop_probability": f"{probability:.8f}",
                            "predicted_label": "DROP" if probability >= 0.5 else "KEEP",
                        }
                    )
    print(f"span DROP probability 저장: {args.output}")


if __name__ == "__main__":
    main()
