"""Train or run the subword token baseline for Span-vs-Token experiments.

Example::

    python script/token_baseline.py \
      --chunks data/aihub/chunks.csv \
      --labels data/subword_labels.csv.gz \
      --split-manifest data/aihub/split_manifest.csv \
      --tokenizer-name encoder/klue_roberta_base_mean_max/best_model/tokenizer \
      --output-dir runs/token_klue_roberta_base

Prediction example::

    python script/token_baseline.py predict \
      --checkpoint runs/token_klue_roberta_base/best_model \
      --chunks data/aihub/chunks.csv \
      --output runs/aihub/token_predictions.jsonl

The default backbone/tokenizer is ``klue/roberta-base``.  Labels are read at
subword level from ``subword_labels.csv.gz``; the legacy eojeol format remains
supported with ``--label-unit word``.  Training and inference use the same
utterance-centered max-length windows as the Span encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_runtime import load_chunks, write_jsonl
from token_baseline import (
    DEFAULT_TOKEN_MODEL,
    TokenClassificationDataset,
    detect_label_unit,
    load_token_examples,
)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(logits: Any, labels: Any) -> Dict[str, float]:
    import torch

    predictions = logits.argmax(dim=-1)
    mask = labels.ne(-100)
    gold = labels[mask]
    predicted = predictions[mask]
    if gold.numel() == 0:
        return {"accuracy": float("nan"), "drop_precision": 0.0, "drop_recall": 0.0, "drop_f1": 0.0}
    tp = int(((predicted == 1) & (gold == 1)).sum().item())
    fp = int(((predicted == 1) & (gold == 0)).sum().item())
    fn = int(((predicted == 0) & (gold == 1)).sum().item())
    accuracy = float((predicted == gold).float().mean().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "drop_precision": precision,
        "drop_recall": recall,
        "drop_f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n_labeled_subwords": float(gold.numel()),
    }


def _merge_metrics(parts: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    totals = {key: sum(float(part.get(key, 0.0)) for part in parts) for key in ("tp", "fp", "fn")}
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    accuracy_values = [part["accuracy"] for part in parts if not math.isnan(float(part["accuracy"]))]
    total_labeled = sum(float(part.get("n_labeled_subwords", 0.0)) for part in parts)
    correct = sum(float(part["accuracy"]) * float(part.get("n_labeled_subwords", 0.0)) for part in parts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / total_labeled if total_labeled else (sum(accuracy_values) / len(accuracy_values) if accuracy_values else float("nan")),
        "drop_precision": precision,
        "drop_recall": recall,
        "drop_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_labeled_subwords": total_labeled,
    }


def _evaluate(model: Any, loader: Any, device: Any, use_fp16: bool) -> Dict[str, float]:
    import torch

    model.eval()
    parts: List[Dict[str, float]] = []
    losses: List[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
            parts.append(_metrics(output.logits.detach().cpu(), batch["labels"].detach().cpu()))
    metrics = _merge_metrics(parts)
    metrics["loss"] = sum(losses) / len(losses) if losses else float("nan")
    return metrics


def _loader(examples: Sequence[Any], tokenizer: Any, max_length: int, batch_size: int, shuffle: bool) -> Any:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification

    dataset = TokenClassificationDataset(examples, tokenizer, max_length)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def _train_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-unit", choices=("auto", "subword", "word"), default="auto")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_TOKEN_MODEL)
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError("학습에는 torch와 transformers가 필요합니다.") from exc

    _set_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda를 지정했지만 CUDA를 사용할 수 없습니다.")
    device = torch.device(device_name)
    use_fp16 = bool(args.fp16 and device.type == "cuda")

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("token baseline 학습에는 fast tokenizer가 필요합니다.")

    resolved_label_unit = detect_label_unit(args.labels, args.label_unit)
    examples = load_token_examples(
        args.chunks,
        args.labels,
        args.split_manifest,
        seed=args.seed,
        label_unit=resolved_label_unit,
    )
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [example for example in examples if example.split == "validation"]
    test_examples = [example for example in examples if example.split == "test"]
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_validation_examples is not None:
        validation_examples = validation_examples[: args.max_validation_examples]
    if not train_examples or not validation_examples:
        raise ValueError(
            f"train/validation split이 필요합니다: train={len(train_examples)}, "
            f"validation={len(validation_examples)}"
        )

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "KEEP", 1: "DROP"},
        label2id={"KEEP": 0, "DROP": 1},
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        config=config,
    ).to(device)

    train_loader = _loader(train_examples, tokenizer, args.max_length, args.batch_size, True)
    validation_loader = _loader(validation_examples, tokenizer, args.max_length, args.batch_size, False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(args.gradient_accumulation, 1))
    total_updates = max(updates_per_epoch * args.epochs, 1)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    output_dir = args.output_dir
    best_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history: List[Dict[str, Any]] = []
    global_update = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss / max(args.gradient_accumulation, 1)
            scaler.scale(loss).backward()
            should_update = (
                batch_index % max(args.gradient_accumulation, 1) == 0
                or batch_index == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

        metrics = _evaluate(model, validation_loader, device, use_fp16)
        metrics["epoch"] = epoch
        metrics["global_update"] = global_update
        history.append(metrics)
        print(
            f"epoch={epoch} loss={metrics['loss']:.5f} "
            f"accuracy={metrics['accuracy']:.4f} drop_f1={metrics['drop_f1']:.4f}",
            flush=True,
        )
        if metrics["drop_f1"] > best_f1:
            best_f1 = metrics["drop_f1"]
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    summary = {
        "model_name": args.model_name,
        "tokenizer_name": tokenizer_name,
        "label_mapping": {"KEEP": 0, "DROP": 1},
        "chunks": str(args.chunks),
        "labels": str(args.labels),
        "label_unit": resolved_label_unit,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "seed": args.seed,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation": args.gradient_accumulation,
        "device": str(device),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "best_drop_f1": best_f1,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"token baseline 저장: {best_dir}")


def _predict_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="학습된 token baseline의 subword별 DROP 확률을 저장합니다.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)

    chunks = load_chunks(args.chunks)
    sentence_ids = list(chunks)
    if args.max_samples is not None:
        sentence_ids = sentence_ids[: args.max_samples]
    from token_baseline import TokenBaselineCompressor

    compressor = TokenBaselineCompressor(
        args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.device,
        max_length=args.max_length,
    )
    rows = []
    for sentence_id in sentence_ids:
        rows.append({"sentence_id": sentence_id, **compressor.score_text(chunks[sentence_id])})
    write_jsonl(args.output, rows)
    print(f"token DROP probability 저장: {args.output}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        _predict_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        _train_main(sys.argv[2:])
        return
    _train_main()


if __name__ == "__main__":
    main()
