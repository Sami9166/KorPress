"""Run a calibrated end-to-end KorQuAD evaluation for Span and Token.

This entry point deliberately starts from the raw KorQuAD question file and
the current encoder probabilities.  It does not merge, reuse, or post-process
the old ``qa_eval``/``qa_results`` files.  Qwen3-8B is the default reader and
tokenizer. It compresses every candidate context, calibrates actual deletion
rate with the Qwen tokenizer, and by default runs the reader only for the
nearest settings at each target deletion rate. Use ``--all-settings`` only
for an exhaustive diagnostic run. Question-level predictions,
compression/survival statistics, and reader latency are written for the
evaluated settings. Official KorQuAD EM/F1 is intentionally not
reimplemented here; prediction JSON files are emitted for the official
evaluator. By default the run uses a deterministic 1,000-question subset;
pass ``--max-questions 0`` to evaluate every supplied question.

The context used for compression is ``chunks.csv``.  That file is the source
whose eojeol indices the dependency-span records refer to; the context string
inside ``qa_pairs.json`` is used only as QA metadata.  This avoids silently
applying span indices to a differently tokenized string.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from experiment_runtime import (
        TokenBaselineCompressor,
        TokenCounter,
        DEFAULT_QWEN_MODEL,
        build_spans_by_chunk,
        compress_span_chunks,
        compress_token_chunks,
        load_chunks,
        load_span_predictions,
        read_jsonl,
        write_csv,
    )
except ImportError:
    from .experiment_runtime import (
        TokenBaselineCompressor,
        TokenCounter,
        DEFAULT_QWEN_MODEL,
        build_spans_by_chunk,
        compress_span_chunks,
        compress_token_chunks,
        load_chunks,
        load_span_predictions,
        read_jsonl,
        write_csv,
    )


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


def _all_answers(qa: Mapping[str, Any]) -> List[str]:
    """Return every reference answer without changing compact input shape."""
    answers = qa.get("answers")
    if isinstance(answers, list):
        values: List[str] = []
        for answer in answers:
            if isinstance(answer, str):
                text = answer
            elif isinstance(answer, Mapping):
                text = str(answer.get("text", ""))
            else:
                text = ""
            if text:
                values.append(text)
        if values:
            return values
    first = _first_answer(qa)
    return [first] if first else []


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
        qas: List[Dict[str, Any]] = []
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
                    "answers": _all_answers(raw_qa),
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


def limit_qa_passages(
    passages: Sequence[Mapping[str, Any]], max_questions: int
) -> List[Dict[str, Any]]:
    """Keep a deterministic prefix of questions without splitting passage IDs.

    A passage can contain multiple questions.  The last selected passage may
    therefore contain only a prefix of its questions; all selected question
    IDs remain stable across reruns and can be used to build a matching gold
    subset for the official evaluator.
    """
    if max_questions < 0:
        raise ValueError("--max-questions는 0 이상이어야 합니다.")
    if max_questions == 0:
        return [dict(passage) for passage in passages]

    selected: List[Dict[str, Any]] = []
    remaining = max_questions
    for passage in passages:
        if remaining <= 0:
            break
        qas = list(passage["qas"])
        if not qas:
            continue
        selected_passage = dict(passage)
        selected_passage["qas"] = qas[:remaining]
        selected.append(selected_passage)
        remaining -= len(selected_passage["qas"])
    return selected


def question_ids(passages: Sequence[Mapping[str, Any]]) -> List[str]:
    return [
        str(qa["question_id"])
        for passage in passages
        for qa in passage["qas"]
    ]


def _normalize_for_survival(text: str) -> str:
    """Normalize only enough to check literal answer survival in context."""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(normalized.split())


def answer_survives(answer: str, compressed_context: str) -> bool:
    normalized_answer = _normalize_for_survival(answer)
    if not normalized_answer:
        return False
    return normalized_answer in _normalize_for_survival(compressed_context)


def _prompt_text(context: str, question: str) -> str:
    """Return the fixed reader prompt used for every compression method.

    The GPT teacher prompt used to create ``span_labels.csv.gz`` is deliberately
    not reused here: it is a data-generation instruction, whereas this prompt
    is the downstream reader instruction applied to original and compressed
    contexts alike. The explicit Korean-output instruction is shared across
    every method.
    """
    return (
        "Read the following text.\n\n"
        f"{context}\n\n"
        "Now, answer the following question based on the above text. "
        "Only output the answer and do not output any other words. "
        "Answer in Korean.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _format_prompt(tokenizer: Any, context: str, question: str) -> str:
    content = _prompt_text(context, question)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()


def _synchronize_device(device: Any) -> None:
    """Synchronize CUDA before/after timing asynchronous generation kernels."""
    try:
        import torch
    except ImportError:
        return
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def generate_answer_timed(
    model: Any,
    tokenizer: Any,
    device: Any,
    context: str,
    question: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Tuple[str, float]:
    """Generate one answer and return wall-clock reader latency in seconds."""
    _synchronize_device(device)
    started = time.perf_counter()
    answer = generate_answer(
        model,
        tokenizer,
        device,
        context,
        question,
        max_input_tokens,
        max_new_tokens,
    )
    _synchronize_device(device)
    return answer, time.perf_counter() - started


def _latency_stats(values: Sequence[float]) -> Dict[str, float]:
    """Summarize per-question reader latency without hiding the raw rows."""
    if not values:
        raise ValueError("latency 값이 없습니다.")
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "reader_latency_total_s": sum(ordered),
        "reader_latency_mean_s": sum(ordered) / len(ordered),
        "reader_latency_median_s": ordered[len(ordered) // 2]
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2,
        "reader_latency_p95_s": ordered[p95_index],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _official_prediction_map(
    rows: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, str]:
    """Build the ``question_id -> prediction`` JSON expected by KorQuAD."""
    predictions: Dict[str, str] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in predictions:
            raise ValueError(
                "공식 KorQuAD prediction key가 중복되었습니다: "
                f"{question_id}"
            )
        predictions[question_id] = str(row[field])
    return predictions


def _official_original_prediction_map(
    passages: Sequence[Mapping[str, Any]],
    predictions: Mapping[Tuple[str, str], str],
) -> Dict[str, str]:
    rows: List[Dict[str, str]] = []
    for passage in passages:
        passage_id = str(passage["passage_id"])
        for qa in passage["qas"]:
            question_id = str(qa["question_id"])
            rows.append(
                {
                    "question_id": question_id,
                    "prediction": str(predictions[(passage_id, question_id)]),
                }
            )
    return _official_prediction_map(rows, "prediction")


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


CompressionJob = Tuple[
    str,
    str,
    Sequence[Mapping[str, Any]],
    Dict[str, Any],
]


def _actual_qwen_deletion_rate(
    rows: Sequence[Mapping[str, Any]], token_counter: TokenCounter
) -> float:
    """Return deletion rate measured with the shared Qwen tokenizer."""
    original_tokens = sum(token_counter.count(str(row["original"])) for row in rows)
    compressed_tokens = sum(token_counter.count(str(row["compressed"])) for row in rows)
    return 1.0 - compressed_tokens / max(original_tokens, 1)


def _selection_row(
    target: float | str,
    method: str,
    setting: str,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    actual_cr: float,
) -> Dict[str, Any]:
    target_gap = ""
    if target != "":
        target_gap = abs(actual_cr - float(target))
    return {
        "target_deletion_rate": target,
        "method": method,
        "drop_rule": metadata.get("drop_rule", ""),
        "L": metadata.get("L", ""),
        "threshold": metadata.get("threshold", ""),
        "retention_rate": metadata.get("retention_rate", ""),
        "setting": setting,
        "actual_qwen_token_deletion_rate": actual_cr,
        "target_cr_gap": target_gap,
        "n_passages": len(rows),
        "selected_for_qa": True,
    }


def _calibration_pair_rows(
    calibration_jobs: Sequence[CompressionJob],
    token_counter: TokenCounter,
    targets: Sequence[float],
    max_cr_gap: float,
    target_tolerance: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set[Tuple[str, str]]]:
    """Search the full Span×Token candidate grid using contexts only."""
    span_jobs = [job for job in calibration_jobs if job[1] == "Span"]
    token_jobs = [job for job in calibration_jobs if job[1] == "Token"]
    if not span_jobs or not token_jobs:
        raise ValueError("calibration 후보에는 Span과 Token이 모두 필요합니다.")

    actual_by_key = {
        (method, setting): _actual_qwen_deletion_rate(rows, token_counter)
        for setting, method, rows, _ in calibration_jobs
    }
    selected_keys: set[Tuple[str, str]] = set()
    selection_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []

    for raw_target in targets:
        target = float(raw_target)
        candidates: List[Tuple[Tuple[Any, ...], CompressionJob, CompressionJob, float, float]] = []
        for span_job in span_jobs:
            span_cr = actual_by_key[(span_job[1], span_job[0])]
            for token_job in token_jobs:
                token_cr = actual_by_key[(token_job[1], token_job[0])]
                span_gap = abs(span_cr - target)
                token_gap = abs(token_cr - target)
                pair_gap = abs(span_cr - token_cr)
                within = (
                    span_gap <= target_tolerance
                    and token_gap <= target_tolerance
                    and pair_gap <= max_cr_gap
                )
                rank = (
                    0 if within else 1,
                    max(span_gap, token_gap),
                    pair_gap,
                    span_gap + token_gap,
                    span_job[0],
                    token_job[0],
                )
                candidates.append((rank, span_job, token_job, span_cr, token_cr))
                grid_rows.append(
                    {
                        "target_deletion_rate": target,
                        "span_setting": span_job[0],
                        "span_L": span_job[3].get("L", ""),
                        "span_drop_rule": span_job[3].get("drop_rule", ""),
                        "span_threshold": span_job[3].get("threshold", ""),
                        "span_calibration_cr": span_cr,
                        "token_setting": token_job[0],
                        "token_threshold": token_job[3].get("threshold", ""),
                        "token_calibration_cr": token_cr,
                        "span_target_gap": span_gap,
                        "token_target_gap": token_gap,
                        "span_token_cr_gap": pair_gap,
                        "within_target_tolerance": (
                            span_gap <= target_tolerance and token_gap <= target_tolerance
                        ),
                        "within_max_cr_gap": pair_gap <= max_cr_gap,
                        "selected": False,
                    }
                )

        _, span_job, token_job, span_cr, token_cr = min(candidates, key=lambda item: item[0])
        selected_keys.add((span_job[1], span_job[0]))
        selected_keys.add((token_job[1], token_job[0]))
        pair_gap = abs(span_cr - token_cr)
        for job, actual_cr in ((span_job, span_cr), (token_job, token_cr)):
            setting, method, rows, metadata = job
            row = _selection_row(target, method, setting, rows, metadata, actual_cr)
            row.update(
                {
                    "calibration_actual_cr": actual_cr,
                    "calibration_target_tolerance": target_tolerance,
                    "selected_pair_cr_gap": pair_gap,
                    "selected_pair_within_max_cr_gap": pair_gap <= max_cr_gap,
                }
            )
            selection_rows.append(row)
        for row in reversed(grid_rows):
            if (
                row["target_deletion_rate"] == target
                and row["span_setting"] == span_job[0]
                and row["token_setting"] == token_job[0]
            ):
                row["selected"] = True
                break
    return selection_rows, grid_rows, selected_keys


def _select_target_jobs(
    calibration_jobs: Sequence[CompressionJob],
    evaluation_jobs: Sequence[CompressionJob],
    token_counter: TokenCounter,
    targets: Sequence[float],
    max_cr_gap: float,
    target_tolerance: float,
) -> Tuple[List[CompressionJob], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select full Span/Token hyperparameter combinations on calibration text."""
    selection_rows, grid_rows, selected_keys = _calibration_pair_rows(
        calibration_jobs,
        token_counter,
        targets,
        max_cr_gap,
        target_tolerance,
    )
    selected_jobs = [
        job for job in evaluation_jobs if (job[1], job[0]) in selected_keys
    ]
    if not selected_jobs:
        raise ValueError("calibration에서 선택된 QA 후보가 없습니다.")
    return selected_jobs, selection_rows, grid_rows


def _all_job_selection_rows(
    jobs: Sequence[CompressionJob], token_counter: TokenCounter
) -> List[Dict[str, Any]]:
    """Describe every candidate when exhaustive QA mode is requested."""
    return [
        _selection_row(
            "",
            method,
            setting,
            rows,
            metadata,
            _actual_qwen_deletion_rate(rows, token_counter),
        )
        for setting, method, rows, metadata in jobs
    ]


def _evaluate_setting(
    setting: str,
    method: str,
    compressed_rows: Sequence[Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, str],
    original_predictions: Mapping[Tuple[str, str], str],
    original_latencies: Mapping[Tuple[str, str], float],
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
    compressed_latencies: List[float] = []

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
            answers = [str(value) for value in qa.get("answers", [answer])]
            key = (passage_id, question_id)
            if key not in original_predictions:
                raise KeyError(f"원문 QA 예측 캐시가 없습니다: {key}")
            if key not in original_latencies:
                raise KeyError(f"원문 QA latency 캐시가 없습니다: {key}")
            pred_original = original_predictions[key]
            pred_compressed, compressed_latency = generate_answer_timed(
                model,
                tokenizer,
                device,
                compressed,
                question,
                max_input_tokens,
                max_new_tokens,
            )
            compressed_latencies.append(compressed_latency)
            survives = any(answer_survives(value, compressed) for value in answers)
            answer_survival_count += int(survives)
            question_count += 1
            output_qas.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "answers": answers,
                    "answer_survives_compression": survives,
                }
            )
            question_rows.append(
                {
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "question": question,
                    "gold": answer,
                    "gold_answers": answers,
                    "pred_original": pred_original,
                    "pred_compressed": pred_compressed,
                    "reader_latency_s_original": original_latencies[key],
                    "reader_latency_s_compressed": compressed_latency,
                    "answer_survives_compression": survives,
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
    }
    summary.update(_latency_stats(compressed_latencies))
    original_latency_stats = _latency_stats(
        [original_latencies[(str(passage["passage_id"]), str(qa["question_id"]))]
         for passage in passages
         for qa in passage["qas"]]
    )
    for key, value in original_latency_stats.items():
        summary[f"original_{key}"] = value
    summary["reader_latency_mean_ratio_vs_original"] = (
        summary["reader_latency_mean_s"] / original_latency_stats["reader_latency_mean_s"]
        if original_latency_stats["reader_latency_mean_s"]
        else None
    )
    summary["reader_latency_mean_delta_s_vs_original"] = (
        summary["reader_latency_mean_s"] - original_latency_stats["reader_latency_mean_s"]
    )
    return eval_passages, question_rows, summary


def _original_summary(
    passages: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, str],
    predictions: Mapping[Tuple[str, str], str],
    latencies: Mapping[Tuple[str, str], float],
    qwen_model: str,
) -> Dict[str, Any]:
    """Build the no-compression QA baseline for the same test passages."""
    survival_values: List[float] = []
    latency_values: List[float] = []
    empty_predictions = 0
    for passage in passages:
        passage_id = str(passage["passage_id"])
        context = chunks[passage_id]
        for qa in passage["qas"]:
            key = (passage_id, str(qa["question_id"]))
            prediction = predictions[key]
            latency_values.append(float(latencies[key]))
            answer = str(qa["answer"])
            answers = [str(value) for value in qa.get("answers", [answer])]
            survival_values.append(
                float(any(answer_survives(value, context) for value in answers))
            )
            empty_predictions += int(not prediction.strip())

    n_questions = len(survival_values)
    if not n_questions:
        raise ValueError("원문 QA baseline을 계산할 질문이 없습니다.")
    summary = {
        "method": "Original",
        "setting": "Original",
        "n_passages": len(passages),
        "n_questions": n_questions,
        "qwen_token_compression_ratio": 0.0,
        "eojeol_compression_ratio": 0.0,
        "answer_survival_rate": sum(survival_values) / n_questions,
        "empty_compressed_context_rate": 0.0,
        "empty_prediction_rate": empty_predictions / n_questions,
        "L": "",
        "threshold": "",
        "drop_rule": "",
        "retention_rate": "",
        "qwen_model": qwen_model,
    }
    summary.update(_latency_stats(latency_values))
    summary["original_reader_latency_total_s"] = summary["reader_latency_total_s"]
    summary["original_reader_latency_mean_s"] = summary["reader_latency_mean_s"]
    summary["original_reader_latency_median_s"] = summary["reader_latency_median_s"]
    summary["original_reader_latency_p95_s"] = summary["reader_latency_p95_s"]
    summary["reader_latency_mean_ratio_vs_original"] = 1.0
    summary["reader_latency_mean_delta_s_vs_original"] = 0.0
    return summary


def _matched_qa(
    summaries: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    max_cr_gap: float,
    selection_rows: Sequence[Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    span = [row for row in summaries if row.get("method") == "Span"]
    token = [row for row in summaries if row.get("method") == "Token"]
    if not span or not token:
        raise ValueError("Span과 Token 결과가 모두 필요합니다.")

    summary_by_key = {
        (str(row.get("method")), str(row.get("setting"))): row
        for row in summaries
    }
    selected_by_target: Dict[Tuple[float, str], Mapping[str, Any]] = {}
    for row in selection_rows or []:
        if row.get("target_deletion_rate", "") == "":
            continue
        selected_by_target[
            (float(row["target_deletion_rate"]), str(row["method"]))
        ] = row

    def fallback_pair(target: float) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        return min(
            ((s, t) for s in span for t in token),
            key=lambda pair: (
                max(
                    abs(float(pair[0]["qwen_token_compression_ratio"]) - target),
                    abs(float(pair[1]["qwen_token_compression_ratio"]) - target),
                ),
                abs(
                    float(pair[0]["qwen_token_compression_ratio"])
                    - float(pair[1]["qwen_token_compression_ratio"])
                ),
            ),
        )

    rows: List[Dict[str, Any]] = []
    for raw_target in targets:
        target = float(raw_target)
        selected_span = selected_by_target.get((target, "Span"))
        selected_token = selected_by_target.get((target, "Token"))
        if selected_span and selected_token:
            span_row = summary_by_key.get(("Span", str(selected_span["setting"])))
            token_row = summary_by_key.get(("Token", str(selected_token["setting"])))
        else:
            span_row, token_row = fallback_pair(target)
        if span_row is None or token_row is None:
            raise ValueError(f"target={target}에 대응하는 QA summary가 없습니다.")

        span_cr = float(span_row["qwen_token_compression_ratio"])
        token_cr = float(token_row["qwen_token_compression_ratio"])
        span_target_gap = abs(span_cr - target)
        token_target_gap = abs(token_cr - target)
        span_token_gap = abs(span_cr - token_cr)
        row: Dict[str, Any] = {
            "target_deletion_rate": target,
            "span_drop_rule": span_row.get("drop_rule", ""),
            "span_L": span_row.get("L", ""),
            "span_threshold": span_row.get("threshold", ""),
            "span_setting": span_row["setting"],
            "span_actual_cr": span_cr,
            "span_cr_gap": span_target_gap,
            "token_setting": token_row["setting"],
            "token_threshold": token_row.get("threshold", ""),
            "token_actual_cr": token_cr,
            "token_cr_gap": token_target_gap,
            "span_token_cr_gap": span_token_gap,
            "max_cr_gap": max_cr_gap,
            "pair_within_max_gap": span_token_gap <= max_cr_gap,
            "span_calibration_cr": (
                selected_span.get("calibration_actual_cr", "")
                if selected_span
                else ""
            ),
            "token_calibration_cr": (
                selected_token.get("calibration_actual_cr", "")
                if selected_token
                else ""
            ),
        }
        for metric in (
            "answer_survival_rate",
            "empty_prediction_rate",
            "reader_latency_mean_s",
            "reader_latency_median_s",
            "reader_latency_p95_s",
            "reader_latency_mean_ratio_vs_original",
            "reader_latency_mean_delta_s_vs_original",
        ):
            s_raw = span_row.get(metric)
            t_raw = token_row.get(metric)
            s_value = "" if s_raw in (None, "") else float(s_raw)
            t_value = "" if t_raw in (None, "") else float(t_raw)
            row[f"span_{metric}"] = s_value
            row[f"token_{metric}"] = t_value
            row[f"delta_{metric}_span_minus_token"] = (
                "" if "" in (s_value, t_value) else s_value - t_value
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--qa-pairs", type=Path, required=True)
    parser.add_argument("--span-records", type=Path, required=True)
    parser.add_argument("--span-predictions", type=Path, required=True)
    parser.add_argument("--calibration-chunks", type=Path)
    parser.add_argument("--calibration-span-records", type=Path)
    parser.add_argument("--calibration-span-predictions", type=Path)
    parser.add_argument(
        "--qwen-model",
        "--qa-model",
        "--qwen-tokenizer",
        dest="qwen_model",
        default=DEFAULT_QWEN_MODEL,
        help="QA reader와 CR tokenizer로 함께 사용할 모델 (기본값: Qwen/Qwen3-8B)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=1000,
        help="평가할 질문 수. 입력 순서의 고정 prefix를 사용하며 0이면 전체 질문입니다.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Reader answer budget; the same value is used for original/compressed contexts.",
    )
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
    parser.add_argument("--span-L", type=int, nargs="+", default=(1, 2, 4, 8), dest="span_lengths")
    parser.add_argument("--span-thresholds", type=float, nargs="+", default=(0.5, 0.7, 0.9))
    parser.add_argument(
        "--drop-rules",
        nargs="+",
        choices=("max", "mean", "min"),
        default=("max", "mean", "min"),
        help="평가할 Span 점수 집계 규칙",
    )
    parser.add_argument("--targets", type=float, nargs="+", default=(0.10, 0.20, 0.30))
    parser.add_argument(
        "--max-cr-gap",
        type=float,
        default=0.02,
        help="Span과 Token 사이에 허용할 실제 삭제율 차이",
    )
    parser.add_argument(
        "--target-cr-tolerance",
        type=float,
        default=0.02,
        help="각 후보가 목표 삭제율에 들어왔다고 볼 허용 오차",
    )
    parser.add_argument(
        "--all-settings",
        action="store_true",
        help="target별 CR에 가까운 설정만 고르지 않고 모든 조합을 QA 평가합니다.",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    passages = limit_qa_passages(load_qa_passages(args.qa_pairs), args.max_questions)
    if not passages or not question_ids(passages):
        raise ValueError("평가할 QA 질문이 없습니다.")
    passage_ids = [str(passage["passage_id"]) for passage in passages]
    missing = [passage_id for passage_id in passage_ids if passage_id not in chunks]
    if missing:
        raise KeyError(f"chunks.csv에 없는 QA passage가 있습니다: {missing[:5]}")

    records = read_jsonl(args.span_records)
    predictions = load_span_predictions(args.span_predictions)
    spans_by_chunk = build_spans_by_chunk(records, predictions, chunks)
    calibration_args = (
        args.calibration_chunks,
        args.calibration_span_records,
        args.calibration_span_predictions,
    )
    if any(value is not None for value in calibration_args) and not all(
        value is not None for value in calibration_args
    ):
        raise ValueError(
            "calibration을 지정할 때는 --calibration-chunks, "
            "--calibration-span-records, --calibration-span-predictions를 "
            "모두 지정해야 합니다."
        )
    calibration_chunks_path = args.calibration_chunks or args.chunks
    calibration_records_path = args.calibration_span_records or args.span_records
    calibration_predictions_path = args.calibration_span_predictions or args.span_predictions
    calibration_chunks = load_chunks(calibration_chunks_path)
    calibration_records = read_jsonl(calibration_records_path)
    calibration_predictions = load_span_predictions(calibration_predictions_path)
    calibration_spans_by_chunk = build_spans_by_chunk(
        calibration_records,
        calibration_predictions,
        calibration_chunks,
    )
    if args.calibration_chunks is None:
        print(
            "주의: 별도 calibration 문맥이 없어 QA 문맥을 calibration에도 사용합니다.",
            flush=True,
        )
    token_counter = TokenCounter(args.qwen_model)

    # Compress first and release each auxiliary encoder before loading the QA
    # reader. This matters on limited-GPU environments when the reader is larger.
    evaluation_jobs: List[CompressionJob] = []
    calibration_jobs: List[CompressionJob] = []
    calibration_ids = list(calibration_chunks)
    token = TokenBaselineCompressor(
        args.token_checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.token_device,
        max_length=args.token_max_length,
    )
    for threshold in args.token_thresholds:
        setting = f"Token_t{threshold:g}"
        metadata = {
            "L": "",
            "threshold": threshold,
            "drop_rule": "",
            "retention_rate": "",
        }
        calibration_jobs.append(
            (
                setting,
                "Token",
                compress_token_chunks(
                    calibration_chunks,
                    token,
                    threshold=threshold,
                    sentence_ids=calibration_ids,
                ),
                metadata,
            )
        )
        evaluation_jobs.append(
            (
                setting,
                "Token",
                compress_token_chunks(
                    chunks,
                    token,
                    threshold=threshold,
                    sentence_ids=passage_ids,
                ),
                metadata,
            )
        )
    del token
    _release_model_memory()

    for drop_rule in args.drop_rules:
        for length in args.span_lengths:
            for threshold in args.span_thresholds:
                setting = f"Span_{drop_rule}_L{length}_t{threshold:g}"
                metadata = {
                    "L": length,
                    "threshold": threshold,
                    "drop_rule": drop_rule,
                    "retention_rate": "",
                }
                calibration_jobs.append(
                    (
                        setting,
                        "Span",
                        compress_span_chunks(
                            calibration_chunks,
                            calibration_spans_by_chunk,
                            L=length,
                            threshold=threshold,
                            drop_rule=drop_rule,
                            sentence_ids=calibration_ids,
                        ),
                        metadata,
                    )
                )
                evaluation_jobs.append(
                    (
                        setting,
                        "Span",
                        compress_span_chunks(
                            chunks,
                            spans_by_chunk,
                            L=length,
                            threshold=threshold,
                            drop_rule=drop_rule,
                            sentence_ids=passage_ids,
                        ),
                        metadata,
                    )
                )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "qa_question_ids.json",
        {
            "max_questions": args.max_questions,
            "n_questions": len(question_ids(passages)),
            "question_ids": question_ids(passages),
        },
    )
    if args.all_settings:
        selected_jobs = list(evaluation_jobs)
        selection_rows = _all_job_selection_rows(evaluation_jobs, token_counter)
        calibration_grid: List[Dict[str, Any]] = []
    else:
        selected_jobs, selection_rows, calibration_grid = _select_target_jobs(
            calibration_jobs,
            evaluation_jobs,
            token_counter,
            args.targets,
            args.max_cr_gap,
            args.target_cr_tolerance,
        )
    write_csv(output_dir / "qa_selected_settings.csv", selection_rows)
    if calibration_grid:
        write_csv(output_dir / "qa_calibration_grid.csv", calibration_grid)
    print(
        f"QA 압축 후보 {len(evaluation_jobs)}개 중 {len(selected_jobs)}개 설정을 reader 평가합니다.",
        flush=True,
    )

    model, tokenizer, device = _load_qa_model(
        args.qwen_model,
        args.device,
        args.torch_dtype,
    )

    original_predictions: Dict[Tuple[str, str], str] = {}
    original_latencies: Dict[Tuple[str, str], float] = {}
    for passage in passages:
        passage_id = str(passage["passage_id"])
        for qa in passage["qas"]:
            question_id = str(qa["question_id"])
            prediction, latency = generate_answer_timed(
                model,
                tokenizer,
                device,
                chunks[passage_id],
                str(qa["question"]),
                args.max_input_tokens,
                args.max_new_tokens,
            )
            key = (passage_id, question_id)
            original_predictions[key] = prediction
            original_latencies[key] = latency

    summaries: List[Dict[str, Any]] = [
        _original_summary(
            passages,
            chunks,
            original_predictions,
            original_latencies,
            args.qwen_model,
        )
    ]
    _write_json(output_dir / "qa_original_summary.json", summaries[0])

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
            original_latencies,
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
        summary["qwen_model"] = args.qwen_model
        _write_json(output_dir / f"qa_eval_{setting}.json", eval_passages)
        _write_json(
            output_dir / f"qa_results_{setting}.json",
            {"per_question": question_rows, "summary": summary},
        )
        _write_json(
            output_dir / f"qa_predictions_{setting}.json",
            _official_prediction_map(question_rows, "pred_compressed"),
        )
        summaries.append(summary)

    for setting, method, rows, metadata in selected_jobs:
        print(f"[QA] {method}/{setting} 시작", flush=True)
        save_setting(setting, method, rows, metadata)
        print(f"[QA] {method}/{setting} 완료", flush=True)

    write_csv(output_dir / "qa_summary.csv", summaries)
    write_csv(
        output_dir / "qa_matched_cr.csv",
        _matched_qa(
            summaries,
            args.targets,
            args.max_cr_gap,
            selection_rows,
        ),
    )
    _write_json(
        output_dir / "qa_original_predictions.json",
        {
            f"{passage_id}::{question_id}": prediction
            for (passage_id, question_id), prediction in original_predictions.items()
        },
    )
    _write_json(
        output_dir / "qa_predictions_original.json",
        _official_original_prediction_map(passages, original_predictions),
    )
    _write_json(
        output_dir / "qa_original_latency.json",
        {
            f"{passage_id}::{question_id}": latency
            for (passage_id, question_id), latency in original_latencies.items()
        },
    )
    _write_json(
        output_dir / "run_config.json",
        vars(args),
    )
    print(f"KorQuAD 재평가 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
