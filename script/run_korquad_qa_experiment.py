"""Run a fresh end-to-end KorQuAD evaluation for Span, Token, and LLMLingua-2.

This entry point deliberately starts from the raw KorQuAD question file and
the current encoder probabilities.  It does not merge, reuse, or post-process
the old ``qa_eval``/``qa_results`` files.  For every compression setting it
compresses the context, runs the same causal QA model on the original and
compressed contexts, and writes both question-level and aggregate metrics.

The context used for compression is ``chunks.csv``.  That file is the source
whose eojeol indices the dependency-span records refer to; the context string
inside ``qa_pairs.json`` is used only as QA metadata.  This avoids silently
applying span indices to a differently tokenized string.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from experiment_runtime import (
        LLMLingua2Compressor,
        TokenBaselineCompressor,
        TokenCounter,
        build_spans_by_chunk,
        compress_ll2_chunks,
        compress_span_chunks,
        compress_token_chunks,
        load_chunks,
        load_span_predictions,
        read_jsonl,
        write_csv,
    )
except ImportError:
    from .experiment_runtime import (
        LLMLingua2Compressor,
        TokenBaselineCompressor,
        TokenCounter,
        build_spans_by_chunk,
        compress_ll2_chunks,
        compress_span_chunks,
        compress_token_chunks,
        load_chunks,
        load_span_predictions,
        read_jsonl,
        write_csv,
    )


PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _first_answer(qa: Mapping[str, Any]) -> str:
    answer = qa.get("answer")
    if isinstance(answer, str):
        return answer
    answers = qa.get("answers")
    if isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, str):
            return first
        if isinstance(first, Mapping):
            return str(first.get("text", ""))
    return ""


def load_qa_passages(path: Path) -> List[Dict[str, Any]]:
    """Load the compact ``passage_id/context/qas`` KorQuAD representation."""
    payload = _read_json(path)
    if isinstance(payload, Mapping):
        payload = payload.get("data", payload.get("passages", []))
    if not isinstance(payload, list):
        raise ValueError(f"QA JSON은 passage 배열이어야 합니다: {path}")

    passages: List[Dict[str, Any]] = []
    for passage in payload:
        if not isinstance(passage, Mapping):
            raise ValueError(f"QA passage 형식이 잘못되었습니다: {path}")
        passage_id = passage.get("passage_id") or passage.get("id")
        context = passage.get("context") or passage.get("paragraph") or ""
        if not passage_id:
            raise ValueError(f"QA passage에 passage_id가 없습니다: {path}")
        raw_qas = passage.get("qas", [])
        if not isinstance(raw_qas, list):
            raise ValueError(f"qas는 배열이어야 합니다: {passage_id}")
        qas: List[Dict[str, str]] = []
        for index, raw_qa in enumerate(raw_qas):
            if not isinstance(raw_qa, Mapping):
                raise ValueError(f"QA 항목 형식이 잘못되었습니다: {passage_id}:{index}")
            question = str(raw_qa.get("question", "")).strip()
            if not question:
                raise ValueError(f"질문이 비어 있습니다: {passage_id}:{index}")
            qas.append(
                {
                    "question_id": str(raw_qa.get("id", f"{passage_id}_q{index}")),
                    "question": question,
                    "answer": _first_answer(raw_qa),
                }
            )
        passages.append(
            {
                "passage_id": str(passage_id),
                "qa_context": str(context),
                "qas": qas,
            }
        )
    if not passages:
        raise ValueError(f"QA passage가 없습니다: {path}")
    return passages


def normalize_answer(text: str) -> str:
    """Korean-friendly SQuAD normalization for EM/F1."""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = PUNCTUATION_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    reference = normalize_answer(gold).split()
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    counts: Dict[str, int] = {}
    for token in predicted:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    remaining = dict(counts)
    for token in reference:
        if remaining.get(token, 0) > 0:
            overlap += 1
            remaining[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def answer_survives(answer: str, compressed_context: str) -> bool:
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    return normalized_answer in normalize_answer(compressed_context)


def _prompt_text(context: str, question: str) -> str:
    return (
        "다음 문맥만 근거로 질문에 대한 정답을 짧게 답하세요. "
        "정답만 출력하고 설명은 쓰지 마세요.\n\n"
        f"[문맥]\n{context}\n\n"
        f"[질문]\n{question}\n\n"
        "[정답]\n"
    )


def _format_prompt(tokenizer: Any, context: str, question: str) -> str:
    content = _prompt_text(context, question)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def _load_qa_model(model_name: str, device_name: str, dtype_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("QA 재평가에는 torch와 transformers가 필요합니다.") from exc

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda를 지정했지만 CUDA를 사용할 수 없습니다.")
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("QA tokenizer에 pad/eos token이 없습니다.")
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {}
    if dtype_name != "auto":
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_answer(
    model: Any,
    tokenizer: Any,
    device: Any,
    context: str,
    question: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    import torch

    prompt = _format_prompt(tokenizer, context, question)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )
    input_length = encoded["input_ids"].shape[-1]
    if input_length > max_input_tokens:
        raise ValueError(
            f"QA prompt가 max_input_tokens를 초과했습니다: "
            f"{input_length} > {max_input_tokens}. "
            "문맥을 임의로 자르지 않도록 max_input_tokens를 늘리세요."
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_kwargs)
    if getattr(model.config, "is_encoder_decoder", False):
        answer_tokens = generated[0]
    else:
        answer_tokens = generated[0, input_length:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _release_model_memory() -> None:
    """Release cached model allocations before loading the QA reader."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_setting(
    setting: str,
    method: str,
    compressed_rows: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, str],
    original_predictions: Mapping[Tuple[str, str], str],
    model: Any,
    tokenizer: Any,
    device: Any,
    token_counter: TokenCounter,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    compressed_by_id = {
        str(row["sentence_id"]): row for row in compressed_rows
    }
    eval_passages: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    original_tokens = 0
    compressed_tokens = 0
    compressed_empty = 0
    answer_survival_count = 0
    question_count = 0

    for passage in passages:
        passage_id = str(passage["passage_id"])
        if passage_id not in chunks:
            raise KeyError(f"chunks.csv에 QA passage가 없습니다: {passage_id}")
        if passage_id not in compressed_by_id:
            raise KeyError(f"압축 결과에 QA passage가 없습니다: {passage_id}")
        original = chunks[passage_id]
        compressed = str(compressed_by_id[passage_id]["compressed"])
        original_count = token_counter.count(original)
        compressed_count = token_counter.count(compressed)
        original_tokens += original_count
        compressed_tokens += compressed_count
        if not compressed.strip():
            compressed_empty += 1

        output_qas: List[Dict[str, Any]] = []
        for qa in passage["qas"]:
            question_id = str(qa["question_id"])
            question = str(qa["question"])
            answer = str(qa["answer"])
            key = (passage_id, question_id)
            if key not in original_predictions:
                raise KeyError(f"원문 QA 예측 캐시가 없습니다: {key}")
            pred_original = original_predictions[key]
            pred_compressed = generate_answer(
                model,
                tokenizer,
                device,
                compressed,
                question,
                max_input_tokens,
                max_new_tokens,
            )
            survives = answer_survives(answer, compressed)
            em_original = exact_match(pred_original, answer)
            em_compressed = exact_match(pred_compressed, answer)
            f1_original = token_f1(pred_original, answer)
            f1_compressed = token_f1(pred_compressed, answer)
            answer_survival_count += int(survives)
            question_count += 1
            output_qas.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "answer_survives_compression": survives,
                }
            )
            question_rows.append(
                {
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "question": question,
                    "gold": answer,
                    "pred_original": pred_original,
                    "pred_compressed": pred_compressed,
                    "answer_survives_compression": survives,
                    "em_original": em_original,
                    "em_compressed": em_compressed,
                    "em_delta": int(em_compressed) - int(em_original),
                    "f1_original": f1_original,
                    "f1_compressed": f1_compressed,
                    "f1_delta": f1_compressed - f1_original,
                }
            )
        eval_passages.append(
            {
                "passage_id": passage_id,
                "original": original,
                "compressed": compressed,
                "n_words_original": len(original.split()),
                "n_words_compressed": len(compressed.split()),
                "n_words_dropped": len(original.split()) - len(compressed.split()),
                "eojeol_compression_ratio": 1.0
                - len(compressed.split()) / max(len(original.split()), 1),
                "qwen_token_compression_ratio": 1.0
                - compressed_count / max(original_count, 1),
                "qas": output_qas,
            }
        )

    if not question_rows:
        raise ValueError(f"QA 항목이 없습니다: {setting}")
    summary: Dict[str, Any] = {
        "method": method,
        "setting": setting,
        "n_passages": len(eval_passages),
        "n_questions": len(question_rows),
        "qwen_token_compression_ratio": 1.0
        - compressed_tokens / max(original_tokens, 1),
        "eojeol_compression_ratio": 1.0
        - sum(row["n_words_compressed"] for row in eval_passages)
        / max(sum(row["n_words_original"] for row in eval_passages), 1),
        "answer_survival_rate": answer_survival_count / question_count,
        "empty_compressed_context_rate": compressed_empty / len(eval_passages),
        "empty_prediction_rate": sum(
            not str(row["pred_compressed"]).strip() for row in question_rows
        )
        / len(question_rows),
        "em_original": sum(bool(row["em_original"]) for row in question_rows)
        / len(question_rows),
        "em_compressed": sum(bool(row["em_compressed"]) for row in question_rows)
        / len(question_rows),
        "f1_original": sum(float(row["f1_original"]) for row in question_rows)
        / len(question_rows),
        "f1_compressed": sum(float(row["f1_compressed"]) for row in question_rows)
        / len(question_rows),
    }
    summary["em_delta"] = summary["em_compressed"] - summary["em_original"]
    summary["f1_delta"] = summary["f1_compressed"] - summary["f1_original"]
    return eval_passages, question_rows, summary


def _matched_qa(
    summaries: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    max_cr_gap: float,
) -> List[Dict[str, Any]]:
    span = [row for row in summaries if row.get("method") == "Span"]
    token = [row for row in summaries if row.get("method") == "Token"]
    ll2 = [row for row in summaries if row.get("method") == "LLMLingua-2"]
    if not span or not token or not ll2:
        raise ValueError("Span, Token, LLMLingua-2 결과가 모두 필요합니다.")

    rows: List[Dict[str, Any]] = []
    for target in targets:
        selected = {}
        for prefix, candidates in (
            ("span", span),
            ("token", token),
            ("ll2", ll2),
        ):
            selected[prefix] = min(
                candidates,
                key=lambda row: abs(
                    float(row["qwen_token_compression_ratio"]) - target
                ),
            )
        span_row = selected["span"]
        token_row = selected["token"]
        ll2_row = selected["ll2"]
        row: Dict[str, Any] = {
            "target_deletion_rate": target,
            "span_setting": span_row["setting"],
            "span_actual_cr": span_row["qwen_token_compression_ratio"],
            "span_cr_gap": abs(
                float(span_row["qwen_token_compression_ratio"]) - target
            ),
            "token_setting": token_row["setting"],
            "token_actual_cr": token_row["qwen_token_compression_ratio"],
            "token_cr_gap": abs(
                float(token_row["qwen_token_compression_ratio"]) - target
            ),
            "ll2_setting": ll2_row["setting"],
            "ll2_actual_cr": ll2_row["qwen_token_compression_ratio"],
            "ll2_cr_gap": abs(
                float(ll2_row["qwen_token_compression_ratio"]) - target
            ),
        }
        row["max_cr_gap"] = max_cr_gap
        row["pair_within_max_gap"] = (
            row["span_cr_gap"] <= max_cr_gap
            and row["token_cr_gap"] <= max_cr_gap
            and row["ll2_cr_gap"] <= max_cr_gap
        )
        for metric in (
            "answer_survival_rate",
            "em_compressed",
            "f1_compressed",
            "em_delta",
            "f1_delta",
            "empty_prediction_rate",
        ):
            s_value = float(span_row[metric])
            t_value = float(token_row[metric])
            l_value = float(ll2_row[metric])
            row[f"span_{metric}"] = s_value
            row[f"token_{metric}"] = t_value
            row[f"ll2_{metric}"] = l_value
            row[f"delta_{metric}_span_minus_token"] = s_value - t_value
            row[f"delta_{metric}_span_minus_ll2"] = s_value - l_value
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--qa-pairs", type=Path, required=True)
    parser.add_argument("--span-records", type=Path, required=True)
    parser.add_argument("--span-predictions", type=Path, required=True)
    parser.add_argument("--qa-model", required=True)
    parser.add_argument("--qwen-tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--token-checkpoint",
        required=True,
        help="학습한 token-level encoder checkpoint 경로",
    )
    parser.add_argument(
        "--tokenizer",
        default="klue/roberta-base",
        help="Token baseline encoder tokenizer (기본값: klue/roberta-base)",
    )
    parser.add_argument(
        "--token-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--token-max-length", type=int, default=512)
    parser.add_argument(
        "--token-thresholds",
        type=float,
        nargs="+",
        default=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        help="Token encoder의 DROP 확률 threshold 목록",
    )
    parser.add_argument("--ll2-model", default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    parser.add_argument("--span-L", type=int, nargs="+", default=(1, 2, 4, 8), dest="span_lengths")
    parser.add_argument("--span-thresholds", type=float, nargs="+", default=(0.5, 0.7, 0.9))
    parser.add_argument("--drop-rules", nargs="+", choices=("max", "mean", "min"), default=("max",))
    parser.add_argument(
        "--ll2-retention-rates",
        type=float,
        nargs="+",
        default=(0.90, 0.78, 0.65),
        help="LLMLingua-2 rate 인자(보존율). 삭제율이 아니라 retention rate입니다.",
    )
    parser.add_argument("--targets", type=float, nargs="+", default=(0.10, 0.20, 0.30))
    parser.add_argument(
        "--max-cr-gap",
        type=float,
        default=0.05,
        help="matched CR 표에 허용할 목표 삭제율과의 최대 차이",
    )
    parser.add_argument("--force-reserve-digit", action="store_true")
    parser.add_argument("--max-samples", type=int, help="개발용 passage 상한. 본 실험에서는 생략하세요.")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    passages = load_qa_passages(args.qa_pairs)
    if args.max_samples is not None:
        passages = passages[: args.max_samples]
    passage_ids = [str(passage["passage_id"]) for passage in passages]
    missing = [passage_id for passage_id in passage_ids if passage_id not in chunks]
    if missing:
        raise KeyError(f"chunks.csv에 없는 QA passage가 있습니다: {missing[:5]}")

    records = read_jsonl(args.span_records)
    predictions = load_span_predictions(args.span_predictions)
    spans_by_chunk = build_spans_by_chunk(records, predictions, chunks)
    token_counter = TokenCounter(args.qwen_tokenizer)

    # Compress first and release each auxiliary encoder before loading the QA
    # reader. This matters on limited-GPU environments when the reader is larger.
    token_jobs: List[Tuple[str, Sequence[Mapping[str, Any]], Dict[str, Any]]] = []
    token = TokenBaselineCompressor(
        args.token_checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.token_device,
        max_length=args.token_max_length,
    )
    for threshold in args.token_thresholds:
        setting = f"Token_t{threshold:g}"
        token_jobs.append(
            (
                setting,
                compress_token_chunks(
                    chunks,
                    token,
                    threshold=threshold,
                    sentence_ids=passage_ids,
                ),
                {
                    "L": "",
                    "threshold": threshold,
                    "drop_rule": "",
                    "retention_rate": "",
                },
            )
        )
    del token
    _release_model_memory()

    ll2_jobs: List[Tuple[str, Sequence[Mapping[str, Any]], Dict[str, Any]]] = []
    ll2 = LLMLingua2Compressor(args.ll2_model, args.force_reserve_digit)
    for retention_rate in args.ll2_retention_rates:
        setting = f"LLMLingua2_r{retention_rate:g}"
        ll2_jobs.append(
            (
                setting,
                compress_ll2_chunks(
                    chunks,
                    ll2,
                    retention_rate=retention_rate,
                    sentence_ids=passage_ids,
                ),
                {
                    "L": "",
                    "threshold": "",
                    "drop_rule": "",
                    "retention_rate": retention_rate,
                },
            )
        )
    del ll2
    _release_model_memory()

    model, tokenizer, device = _load_qa_model(
        args.qa_model,
        args.device,
        args.torch_dtype,
    )

    original_predictions: Dict[Tuple[str, str], str] = {}
    for passage in passages:
        passage_id = str(passage["passage_id"])
        for qa in passage["qas"]:
            question_id = str(qa["question_id"])
            original_predictions[(passage_id, question_id)] = generate_answer(
                model,
                tokenizer,
                device,
                chunks[passage_id],
                str(qa["question"]),
                args.max_input_tokens,
                args.max_new_tokens,
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []

    def save_setting(
        setting: str,
        method: str,
        rows: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        eval_passages, question_rows, summary = _evaluate_setting(
            setting,
            method,
            rows,
            passages,
            chunks,
            original_predictions,
            model,
            tokenizer,
            device,
            token_counter,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        for item in eval_passages:
            item["method"] = method
            item["setting"] = setting
        for item in question_rows:
            item["method"] = method
            item["setting"] = setting
        if metadata:
            summary.update(metadata)
        _write_json(output_dir / f"qa_eval_{setting}.json", eval_passages)
        _write_json(
            output_dir / f"qa_results_{setting}.json",
            {"per_question": question_rows, "summary": summary},
        )
        summaries.append(summary)

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
                    sentence_ids=passage_ids,
                )
                save_setting(
                    setting,
                    "Span",
                    rows,
                    {
                        "L": length,
                        "threshold": threshold,
                        "drop_rule": drop_rule,
                        "retention_rate": "",
                    },
                )

    for setting, rows, metadata in token_jobs:
        save_setting(
            setting,
            "Token",
            rows,
            metadata,
        )

    for setting, rows, metadata in ll2_jobs:
        save_setting(
            setting,
            "LLMLingua-2",
            rows,
            metadata,
        )

    write_csv(output_dir / "qa_summary.csv", summaries)
    write_csv(
        output_dir / "qa_matched_cr.csv",
        _matched_qa(summaries, args.targets, args.max_cr_gap),
    )
    _write_json(
        output_dir / "qa_original_predictions.json",
        {
            f"{passage_id}::{question_id}": prediction
            for (passage_id, question_id), prediction in original_predictions.items()
        },
    )
    _write_json(
        output_dir / "run_config.json",
        vars(args),
    )
    print(f"KorQuAD 재평가 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
