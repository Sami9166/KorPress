"""Measure the effect of Span compressor hyperparameters.

This experiment intentionally evaluates Span only. It sweeps ``L``,
``drop_rule``, and ``threshold`` on the split supplied by ``--span-records``
and writes one row per setting. A small target-rate table is also written as
a convenience for choosing representative settings; it is not a QA result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from experiment_runtime import (
        DEFAULT_QWEN_MODEL,
        TokenCounter,
        build_spans_by_chunk,
        compress_span_chunks,
        intrinsic_summary,
        load_chunks,
        load_span_predictions,
        read_jsonl,
        write_csv,
        write_jsonl,
    )
except ImportError:
    from .experiment_runtime import (
        DEFAULT_QWEN_MODEL,
        TokenCounter,
        build_spans_by_chunk,
        compress_span_chunks,
        intrinsic_summary,
        load_chunks,
        load_span_predictions,
        read_jsonl,
        write_csv,
        write_jsonl,
    )


def _semantic_model(name: str | None):
    if not name or name.lower() in {"none", "off"}:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "semantic similarity를 계산하려면 sentence-transformers가 필요합니다."
        ) from exc
    return SentenceTransformer(name)


def _semantic_value(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("semantic_mean", ""))
    except (TypeError, ValueError):
        return float("-inf")


def _target_examples(
    summaries: Sequence[Mapping[str, Any]], targets: Sequence[float]
) -> List[Dict[str, Any]]:
    """Return representative rows without hiding the complete sweep."""
    result: List[Dict[str, Any]] = []
    for target in targets:
        row = min(
            summaries,
            key=lambda item: (
                abs(float(item["qwen_token_compression_ratio"]) - float(target)),
                -_semantic_value(item),
                str(item.get("setting", "")),
            ),
        )
        result.append(
            {
                "target_deletion_rate": float(target),
                "setting": row["setting"],
                "L": row.get("L", ""),
                "drop_rule": row.get("drop_rule", ""),
                "threshold": row.get("threshold", ""),
                "actual_deletion_rate": row["qwen_token_compression_ratio"],
                "target_cr_gap": abs(
                    float(row["qwen_token_compression_ratio"]) - float(target)
                ),
                "semantic_mean": row.get("semantic_mean", ""),
                "number_retention": row.get("number_retention", ""),
                "date_retention": row.get("date_retention", ""),
                "negation_retention": row.get("negation_retention", ""),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--span-records", type=Path, required=True)
    parser.add_argument("--span-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qwen-model",
        "--qwen-tokenizer",
        dest="qwen_model",
        default=DEFAULT_QWEN_MODEL,
        help="실제 CR 계산에 사용할 tokenizer 모델",
    )
    parser.add_argument(
        "--similarity-model",
        default="dragonkue/BGE-m3-ko",
        help="의미 유사도 모델. 끄려면 none",
    )
    parser.add_argument(
        "--span-L",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
        dest="span_lengths",
    )
    parser.add_argument(
        "--span-thresholds",
        type=float,
        nargs="+",
        default=(0.5, 0.7, 0.9),
    )
    parser.add_argument(
        "--drop-rules",
        nargs="+",
        choices=("max", "mean", "min"),
        default=("max", "mean", "min"),
    )
    parser.add_argument("--targets", type=float, nargs="+", default=(0.10, 0.20, 0.30))
    parser.add_argument("--max-samples", type=int, help="개발용 평가 표본 상한")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    records = read_jsonl(args.span_records)
    predictions = load_span_predictions(args.span_predictions)
    spans_by_chunk = build_spans_by_chunk(records, predictions, chunks)
    sentence_ids = [sentence_id for sentence_id in chunks if sentence_id in spans_by_chunk]
    if not sentence_ids:
        raise ValueError("span-records와 chunks 사이에 평가할 sentence_id가 없습니다.")
    if args.max_samples is not None:
        sentence_ids = sentence_ids[: args.max_samples]

    token_counter = TokenCounter(args.qwen_model)
    semantic_model = _semantic_model(args.similarity_model)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []

    for drop_rule in args.drop_rules:
        for length in args.span_lengths:
            for threshold in args.span_thresholds:
                setting = f"Span_{drop_rule}_L{length}_t{threshold:g}"
                rows = compress_span_chunks(
                    chunks,
                    spans_by_chunk,
                    L=length,
                    threshold=threshold,
                    drop_rule=drop_rule,
                    sentence_ids=sentence_ids,
                )
                for row in rows:
                    row["setting"] = setting
                write_jsonl(output_dir / f"{setting}.jsonl", rows)
                summary = intrinsic_summary(rows, token_counter, semantic_model)
                summary.update(
                    {
                        "setting": setting,
                        "L": length,
                        "threshold": threshold,
                        "drop_rule": drop_rule,
                        "qwen_model": args.qwen_model,
                    }
                )
                summaries.append(summary)

    write_csv(output_dir / "intrinsic_summary.csv", summaries)
    write_csv(
        output_dir / "intrinsic_selected_settings.csv",
        _target_examples(summaries, args.targets),
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Span intrinsic 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
