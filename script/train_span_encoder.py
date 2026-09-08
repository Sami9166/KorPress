"""Fine-tune KLUE-RoBERTa with interruption-safe checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from span_encoder_data import IndexedJsonlDataset, SpanBatchCollator, load_chunk_words, move_batch_to_device
from span_encoder_model import ContextualSpanClassifier, SpanEncoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Persistent directory for last_training_state.pt; defaults to output-dir",
    )
    parser.add_argument(
        "--resume-from",
        default="none",
        help="none, auto, or an explicit training-state .pt path",
    )
    parser.add_argument("--save-every-updates", type=int, default=250)
    parser.add_argument(
        "--delete-resume-checkpoint-on-complete",
        action="store_true",
        help="Delete the large optimizer resume state after successful completion",
    )
    parser.add_argument("--model-name", default="klue/roberta-base")
    parser.add_argument("--pooling", choices=["mean", "max", "mean_max"], default="mean_max")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2, help="Number of target utterances")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-utterances", type=int, default=0)
    parser.add_argument("--max-validation-utterances", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpochSliceSampler(Sampler[int]):
    """Deterministic epoch shuffle that can restart at an exact example offset."""

    def __init__(self, size: int, seed: int, epoch: int, start_index: int = 0) -> None:
        self.size = size
        self.start_index = min(start_index, size)
        generator = torch.Generator().manual_seed(seed + epoch)
        self.order = torch.randperm(size, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.order[self.start_index :])

    def __len__(self) -> int:
        return self.size - self.start_index


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "drop_precision": precision,
        "drop_recall": recall,
        "drop_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "span_count": total,
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device, threshold: float) -> dict:
    model.eval()
    loss_sum = 0.0
    span_count = 0
    tp = fp = fn = tn = 0
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        output = model(**batch)
        labels = batch["labels"]
        predictions = (output["drop_probabilities"] >= threshold).long()
        count = labels.numel()
        loss_sum += output["loss"].item() * count
        span_count += count
        tp += int(((predictions == 1) & (labels == 1)).sum())
        fp += int(((predictions == 1) & (labels == 0)).sum())
        fn += int(((predictions == 0) & (labels == 1)).sum())
        tn += int(((predictions == 0) & (labels == 0)).sum())
    metrics = metrics_from_counts(tp, fp, fn, tn)
    metrics["loss"] = loss_sum / span_count if span_count else math.nan
    return metrics


def atomic_json_dump(payload: dict, path: Path) -> None:
    """Best-effort progress write.

    Google Drive FUSE can make a shared ``.tmp`` path disappear during rename,
    especially when two runtimes touch the same run directory. Progress JSON is
    diagnostic only, so its failure must never terminate model training.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    last_error: OSError | None = None
    for attempt in range(1, 4):
        try:
            path.write_text(serialized, encoding="utf-8")
            return
        except OSError as error:
            last_error = error
            print(
                f"Warning: progress write failed ({attempt}/3): {error}",
                flush=True,
            )
            time.sleep(2)
    print(
        f"Warning: continuing without updating {path}: {last_error}",
        flush=True,
    )


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def training_state_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch: int,
    next_batch: int,
    global_update: int,
    best_f1: float,
    history: list[dict],
) -> dict:
    return {
        "format_version": 1,
        "span_config": asdict(model.span_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_update": global_update,
        "best_f1": best_f1,
        "history": history,
        "rng_state": capture_rng_state(),
        "critical_args": {
            "model_name": args.model_name,
            "pooling": args.pooling,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "epochs": args.epochs,
            "seed": args.seed,
            "max_train_utterances": args.max_train_utterances,
        },
    }


def resolve_resume_path(value: str, checkpoint_dir: Path) -> Path | None:
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        candidate = checkpoint_dir / "last_training_state.pt"
        return candidate if candidate.exists() else None
    candidate = Path(value)
    if not candidate.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    return candidate


def remove_stale_checkpoint_temporary(checkpoint_dir: Path) -> None:
    """An interrupted atomic save may leave a large .tmp file in Drive."""
    temporary = checkpoint_dir / "last_training_state.pt.tmp"
    if temporary.exists():
        temporary.unlink()
        print(f"Removed stale temporary checkpoint: {temporary}")


def validate_resume_args(saved: dict, args: argparse.Namespace) -> None:
    current = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "epochs": args.epochs,
        "seed": args.seed,
        "max_train_utterances": args.max_train_utterances,
    }
    mismatches = {key: (saved.get(key), value) for key, value in current.items() if saved.get(key) != value}
    if mismatches:
        raise ValueError(f"Resume arguments differ from checkpoint: {mismatches}")


def limited_dataset(dataset, limit: int):
    return Subset(dataset, range(min(limit, len(dataset)))) if limit else dataset


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or args.output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_checkpoint_temporary(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    chunk_words = load_chunk_words(args.chunks)
    collator = SpanBatchCollator(tokenizer, chunk_words, max_length=args.max_length)
    train_data = limited_dataset(
        IndexedJsonlDataset(args.data_dir / "train.jsonl"), args.max_train_utterances
    )
    valid_data = limited_dataset(
        IndexedJsonlDataset(args.data_dir / "validation.jsonl"),
        args.max_validation_utterances,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = SpanEncoderConfig(model_name=args.model_name, pooling=args.pooling)
    model = ContextualSpanClassifier(config).to(device)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    no_decay = {"bias", "LayerNorm.weight"}
    optimizer = AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
    )
    total_batches = math.ceil(len(train_data) / args.batch_size)
    updates_per_epoch = math.ceil(total_batches / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    use_scaler = device.type == "cuda" and args.precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    amp_enabled = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    start_epoch = 1
    start_batch = 0
    global_update = 0
    history: list[dict] = []
    best_f1 = -1.0
    resume_path = resolve_resume_path(args.resume_from, checkpoint_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_args(checkpoint["critical_args"], args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["next_batch"])
        global_update = int(checkpoint["global_update"])
        best_f1 = float(checkpoint["best_f1"])
        history = list(checkpoint["history"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"Resumed from {resume_path}: epoch={start_epoch}, next_batch={start_batch}")
    last_checkpoint_update = global_update

    atomic_json_dump(
        {
            "status": "running",
            "device": str(device),
            "epoch": start_epoch,
            "next_batch": start_batch,
            "total_batches_per_epoch": total_batches,
            "global_update": global_update,
            "resumable_from_update": last_checkpoint_update,
            "total_updates": total_updates,
        },
        checkpoint_dir / "progress.json",
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        sampler = EpochSliceSampler(
            len(train_data),
            args.seed,
            epoch,
            start_index=epoch_start_batch * args.batch_size,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model.train()
        running_loss = 0.0
        running_batches = 0
        accumulation_count = 0
        for batch_index, raw_batch in enumerate(train_loader, start=epoch_start_batch):
            batch = move_batch_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(**batch)
                group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
                accumulation_target = min(
                    args.gradient_accumulation, total_batches - group_start
                )
                loss = output["loss"] / accumulation_target
            scaler.scale(loss).backward()
            running_loss += output["loss"].item()
            running_batches += 1
            accumulation_count += 1
            next_batch = batch_index + 1
            should_update = (
                accumulation_count == accumulation_target
                or next_batch == total_batches
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                global_update += 1

                if args.save_every_updates and global_update % args.save_every_updates == 0:
                    atomic_torch_save(
                        training_state_payload(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            args,
                            epoch,
                            next_batch,
                            global_update,
                            best_f1,
                            history,
                        ),
                        checkpoint_dir / "last_training_state.pt",
                    )
                    last_checkpoint_update = global_update
                    print(f"Saved resumable checkpoint at update={global_update}")

            if running_batches >= args.log_every:
                print(
                    f"epoch={epoch}/{args.epochs} batch={next_batch}/{total_batches} "
                    f"update={global_update}/{total_updates} "
                    f"loss={running_loss / running_batches:.6f}",
                    flush=True,
                )
                running_loss = 0.0
                running_batches = 0
                atomic_json_dump(
                    {
                        "status": "running",
                        "device": str(device),
                        "epoch": epoch,
                        "next_batch": next_batch,
                        "total_batches_per_epoch": total_batches,
                        "global_update": global_update,
                        "resumable_from_update": last_checkpoint_update,
                        "total_updates": total_updates,
                        "best_drop_f1": best_f1,
                    },
                    checkpoint_dir / "progress.json",
                )

        valid_metrics = evaluate(model, valid_loader, device, args.threshold)
        valid_metrics["epoch"] = epoch
        history.append(valid_metrics)
        print(json.dumps(valid_metrics, ensure_ascii=False, indent=2), flush=True)
        if valid_metrics["drop_f1"] > best_f1:
            best_f1 = float(valid_metrics["drop_f1"])
            best_dir = args.output_dir / "best_model"
            model.save_checkpoint(best_dir)
            tokenizer.save_pretrained(best_dir / "tokenizer")

        atomic_torch_save(
            training_state_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                epoch + 1,
                0,
                global_update,
                best_f1,
                history,
            ),
            checkpoint_dir / "last_training_state.pt",
        )
        last_checkpoint_update = global_update
        start_batch = 0

    run = {
        "arguments": vars(args),
        "device": str(device),
        "train_utterances": len(train_data),
        "validation_utterances": len(valid_data),
        "best_drop_f1": best_f1,
        "history": history,
    }
    atomic_json_dump(run, args.output_dir / "training_summary.json")
    atomic_json_dump(
        {
            "status": "complete",
            "device": str(device),
            "epoch": args.epochs,
            "global_update": global_update,
            "total_updates": total_updates,
            "best_drop_f1": best_f1,
        },
        checkpoint_dir / "progress.json",
    )
    if args.delete_resume_checkpoint_on_complete:
        resume_checkpoint = checkpoint_dir / "last_training_state.pt"
        if resume_checkpoint.exists():
            resume_checkpoint.unlink()
            print(f"Deleted completed-run resume checkpoint: {resume_checkpoint}")


if __name__ == "__main__":
    main()
