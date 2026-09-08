"""Run a fresh intrinsic Span-vs-Token experiment.

The script starts from raw chunks, current Span encoder probabilities, and a
same-size token classifier. It writes compressed examples, per-setting
metrics, and a matched-deletion-rate table. The official LLMLingua-2 model is
reserved for the KorQuAD QA experiment.

The encoder probabilities are produced with ``predict_span_encoder.py`` from a
local checkpoint. The matched table pairs each Span setting with the Token
setting having the closest *actual* Qwen-token deletion rate.
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
        TokenBaselineCompressor,
        TokenCounter,
        DEFAULT_QWEN_MODEL,
        build_spans_by_chunk,
        closest_rows,
        compress_span_chunks,
        compress_token_chunks,
        intrinsic_summary,
        load_chunks,
        load_span_predictions,
        read_jsonl,
        write_csv,
        write_jsonl,
    )
except ImportError:
    from .experiment_runtime import (
        TokenBaselineCompressor,
        TokenCounter,
        DEFAULT_QWEN_MODEL,
        build_spans_by_chunk,
        closest_rows,
        compress_span_chunks,
        compress_token_chunks,
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


def _set_metadata(rows: List[Dict[str, Any]], setting: str) -> None:
    for row in rows:
        row["setting"] = setting


def _paired_intrinsic(
    summaries: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    max_cr_gap: float,
) -> List[Dict[str, Any]]:
    span = [row for row in summaries if row.get("method") == "Span"]
    token = [row for row in summaries if row.get("method") == "Token"]
    if not span or not token:
        raise ValueError("Span과 Token baseline 결과가 모두 필요합니다.")

    rows: List[Dict[str, Any]] = []
    for selected in closest_rows(span, targets):
        target = float(selected["target_deletion_rate"])
        span_row = min(
            span,
            key=lambda row: (
                abs(float(row["qwen_token_compression_ratio"]) - target),
                -float(row.get("semantic_mean") or float("-inf")),
            ),
        )
        token_row = min(
            token,
            key=lambda row: abs(
                float(row["qwen_token_compression_ratio"])
                - float(span_row["qwen_token_compression_ratio"])
            ),
        )
        span_cr = float(span_row["qwen_token_compression_ratio"])
        token_cr = float(token_row["qwen_token_compression_ratio"])
        cr_gap = abs(span_cr - token_cr)
        row: Dict[str, Any] = {
            "target_deletion_rate": target,
            "span_setting": span_row["setting"],
            "span_actual_cr": span_cr,
            "token_setting": token_row["setting"],
            "token_actual_cr": token_cr,
            "actual_cr_gap": cr_gap,
            "pair_within_max_gap": cr_gap <= max_cr_gap,
        }
        for metric in (
            "number_retention",
            "date_retention",
            "negation_retention",
            "semantic_mean",
        ):
            row[f"span_{metric}"] = span_row.get(metric, "")
            row[f"token_{metric}"] = token_row.get(metric, "")
        rows.append(row)
    return rows


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
        help="CR 계산에 사용할 Qwen tokenizer (기본값: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--similarity-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help=(
            "의미 유사도 모델 (기본값: "
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2; "
            "끄려면 none)"
        ),
    )
    parser.add_argument("--token-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        default="klue/roberta-base",
        help="Span encoder와 동일한 tokenizer를 기본값으로 사용합니다.",
    )
    parser.add_argument("--token-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--token-max-length", type=int, default=512)
    parser.add_argument("--span-L", type=int, nargs="+", default=(1, 2, 4, 8), dest="span_lengths")
    parser.add_argument("--span-thresholds", type=float, nargs="+", default=(0.5, 0.7, 0.9))
    parser.add_argument("--drop-rules", nargs="+", choices=("max", "mean", "min"), default=("max",))
    parser.add_argument(
        "--token-thresholds",
        type=float,
        nargs="+",
        default=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        help="token DROP 확률 threshold 목록",
    )
    parser.add_argument("--targets", type=float, nargs="+", default=(0.10, 0.20, 0.30))
    parser.add_argument(
        "--max-cr-gap",
        type=float,
        default=0.01,
        help="matched CR 표에서 Span/Token 실제 삭제율 차이의 최대 허용값",
    )
    parser.add_argument("--max-samples", type=int, help="개발용 상한. 본 실험에서는 생략하세요.")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    records = read_jsonl(args.span_records)
    predictions = load_span_predictions(args.span_predictions)
    spans_by_chunk = build_spans_by_chunk(records, predictions, chunks)
    # span-records가 지정한 split만 평가한다. 전체 chunks를 사용하면
    # Token은 train/validation까지 압축하고 Span은 record가 없는 chunk를
    # 원문 그대로 두어 split 누수와 방법별 표본 불일치가 생긴다.
    sentence_ids = [sentence_id for sentence_id in chunks if sentence_id in spans_by_chunk]
    if not sentence_ids:
        raise ValueError("span-records와 chunks 사이에 평가할 sentence_id가 없습니다.")
    if args.max_samples is not None:
        sentence_ids = sentence_ids[: args.max_samples]
    token_counter = TokenCounter(args.qwen_model)
    semantic_model = _semantic_model(args.similarity_model)
    token = TokenBaselineCompressor(
        args.token_checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.token_device,
        max_length=args.token_max_length,
    )

    summaries: List[Dict[str, Any]] = []
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
                _set_metadata(rows, setting)
                write_jsonl(output_dir / f"{setting}.jsonl", rows)
                summary = intrinsic_summary(rows, token_counter, semantic_model)
                summary.update(
                    {
                        "setting": setting,
                        "L": length,
                        "threshold": threshold,
                        "drop_rule": drop_rule,
                        "retention_rate": "",
                        "qwen_model": args.qwen_model,
                    }
                )
                summaries.append(summary)

    for threshold in args.token_thresholds:
        setting = f"Token_t{threshold:g}"
        rows = compress_token_chunks(
            chunks,
            token,
            threshold=threshold,
            sentence_ids=sentence_ids,
        )
        _set_metadata(rows, setting)
        write_jsonl(output_dir / f"{setting}.jsonl", rows)
        summary = intrinsic_summary(rows, token_counter, semantic_model)
        summary.update(
            {
                "setting": setting,
                "L": "",
                "threshold": threshold,
                "drop_rule": "",
                "retention_rate": "",
                "qwen_model": args.qwen_model,
            }
        )
        summaries.append(summary)

    write_csv(output_dir / "intrinsic_summary.csv", summaries)
    write_csv(
        output_dir / "intrinsic_matched_cr.csv",
        _paired_intrinsic(summaries, args.targets, args.max_cr_gap),
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"intrinsic 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
