"""Prepare bounded KorQuAD dev inputs for the KorPress QA runner.

The official KorQuAD dev release is a SQuAD-shaped JSON file (or one or more
ZIP shards).  This command can read local dev files or download a Hugging Face
dataset split, selects a reproducible random question sample, splits long
contexts into tokenizer-bounded windows, and writes the compact files consumed
by ``run_korquad_qa_experiment.py``:

With no ``--input`` or ``--hf-dataset``, it downloads
``LGCNS/KorQuAD_2.0``'s ``validation`` split.

    chunks.csv
    qa_pairs.json
    spans.jsonl
    gold_subset.json
    question_ids.json
    manifest.json

``spans.jsonl`` contains inference-only candidates.  The current prediction
CLI requires a ``label`` field for batching, so candidates use ``KEEP`` as a
placeholder; the label is not used by the checkpoint's probabilities.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dependency_spans import (  # noqa: E402
    KoreanDependencyParser,
    download_korean_model,
    generate_spans_up_to_L,
)


DEV_RE = re.compile(r"(?:^|[._\-/ ])dev(?:$|[._\-/ ])", re.IGNORECASE)
WORD_RE = re.compile(r"\S+")


@dataclass
class RawContext:
    source: str
    title: str
    context: str
    qas: list[dict[str, Any]]
    context_index: int


@dataclass
class AnswerInfo:
    text: str
    start: int | None


@dataclass
class QaInfo:
    question_id: str
    question: str
    answers: list[AnswerInfo]
    raw: dict[str, Any]
    context_index: int
    word_interval: tuple[int, int] | None = None


def _is_dev_name(value: str) -> bool:
    lowered = value.lower().replace("\\", "/")
    return bool(DEV_RE.search(lowered)) or "dev" in lowered


def _discover_inputs(inputs: Sequence[Path]) -> list[Path]:
    """Find JSON/ZIP files for the dev split.

    Explicit files are accepted even when their filename does not contain
    ``dev``.  For a directory, only paths that look like dev files are chosen
    so that a parent directory containing train and test is safe to pass.
    """

    found: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"입력 경로가 없습니다: {path}")
        if path.is_file():
            if path.suffix.lower() not in {".json", ".zip"}:
                raise ValueError(f"JSON 또는 ZIP 파일이 아닙니다: {path}")
            found.append(path.resolve())
            continue

        candidates = sorted(
            candidate.resolve()
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in {".json", ".zip"}
            and _is_dev_name(str(candidate))
        )
        if not candidates and _is_dev_name(str(path)):
            candidates = sorted(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".json", ".zip"}
            )
        if not candidates:
            raise FileNotFoundError(
                f"디렉터리에서 dev JSON/ZIP을 찾지 못했습니다: {path}"
            )
        found.extend(candidates)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return sorted(unique)


def _json_payload(handle: io.TextIOBase, source: str) -> Mapping[str, Any]:
    try:
        payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"KorQuAD JSON을 읽지 못했습니다: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"KorQuAD JSON 최상위 객체가 아닙니다: {source}")
    if not isinstance(payload.get("data"), list):
        raise ValueError(f"KorQuAD JSON에 data 배열이 없습니다: {source}")
    return payload


def _iter_payloads(paths: Sequence[Path]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for path in paths:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8-sig") as handle:
                yield str(path), _json_payload(handle, str(path))
            continue

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".json")
            )
            dev_names = [name for name in names if _is_dev_name(name)]
            if dev_names:
                names = dev_names
            elif not _is_dev_name(str(path)):
                raise FileNotFoundError(
                    f"ZIP 안에서 dev JSON을 찾지 못했습니다: {path}"
                )
            if not names:
                raise FileNotFoundError(f"ZIP 안에 JSON이 없습니다: {path}")
            for name in names:
                with archive.open(name) as binary_handle:
                    text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8")
                    try:
                        yield f"{path}!{name}", _json_payload(
                            text_handle, f"{path}!{name}"
                        )
                    finally:
                        text_handle.detach()


def _iter_contexts(
    payload: Mapping[str, Any], source: str, start_index: int
) -> Iterator[RawContext]:
    for article_index, article in enumerate(payload["data"]):
        if not isinstance(article, Mapping):
            continue
        title = str(article.get("title", f"article_{article_index:06d}"))
        if "context" in article:
            qas = article.get("qas", [])
            if isinstance(qas, list):
                yield RawContext(
                    source,
                    title,
                    str(article.get("context", "")),
                    [dict(qa) for qa in qas if isinstance(qa, Mapping)],
                    start_index,
                )
            start_index += 1
            continue

        paragraphs = article.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            continue
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, Mapping):
                continue
            qas = paragraph.get("qas", [])
            if not isinstance(qas, list):
                qas = []
            yield RawContext(
                source,
                title,
                str(paragraph.get("context", "")),
                [dict(qa) for qa in qas if isinstance(qa, Mapping)],
                start_index,
            )
            start_index += 1


def _answer_infos(raw_qa: Mapping[str, Any]) -> list[AnswerInfo]:
    raw_answers = raw_qa.get("answers", [])
    if not isinstance(raw_answers, list) or not raw_answers:
        if raw_qa.get("answer"):
            raw_answers = [{"text": raw_qa["answer"]}]
        else:
            raw_answers = []

    answers: list[AnswerInfo] = []
    for raw_answer in raw_answers:
        if isinstance(raw_answer, Mapping):
            text = str(raw_answer.get("text", ""))
            raw_start = raw_answer.get("answer_start")
        else:
            text = str(raw_answer)
            raw_start = None
        if not text:
            continue
        start: int | None
        try:
            start = int(raw_start) if raw_start is not None else None
        except (TypeError, ValueError):
            start = None
        answers.append(AnswerInfo(text, start))
    return answers


def _collect_questions(contexts: Sequence[RawContext]) -> list[QaInfo]:
    questions: list[QaInfo] = []
    seen: set[str] = set()
    for context in contexts:
        for qa_index, raw_qa in enumerate(context.qas):
            question_id = str(raw_qa.get("id", "")).strip()
            if not question_id:
                question_id = f"context_{context.context_index:06d}_q{qa_index:04d}"
            if question_id in seen:
                raise ValueError(f"질문 ID가 중복됩니다: {question_id}")
            question = str(raw_qa.get("question", "")).strip()
            answers = _answer_infos(raw_qa)
            if not question or not answers:
                continue
            seen.add(question_id)
            raw_copy = dict(raw_qa)
            raw_copy.setdefault("id", question_id)
            questions.append(
                QaInfo(
                    question_id,
                    question,
                    answers,
                    raw_copy,
                    context.context_index,
                )
            )
    return questions


def _hf_answer_list(value: Any) -> list[dict[str, Any]]:
    """Convert the flattened HF answer feature to SQuAD answer objects."""

    if isinstance(value, Mapping):
        texts = value.get("text", [])
        starts = value.get("answer_start", [])
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, Sequence) or isinstance(texts, (bytes, bytearray)):
            texts = []
        if isinstance(starts, (int, float, str)):
            starts = [starts]
        elif not isinstance(starts, Sequence) or isinstance(starts, (bytes, bytearray)):
            starts = []
        answers: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            item: dict[str, Any] = {"text": str(text)}
            if index < len(starts) and starts[index] not in (None, ""):
                try:
                    item["answer_start"] = int(starts[index])
                except (TypeError, ValueError):
                    pass
            answers.append(item)
        return answers
    if isinstance(value, list):
        answers = []
        for item in value:
            if isinstance(item, Mapping):
                answer = {"text": str(item.get("text", ""))}
                if item.get("answer_start") is not None:
                    answer["answer_start"] = item["answer_start"]
                if answer["text"]:
                    answers.append(answer)
            elif item:
                answers.append({"text": str(item)})
        return answers
    return [{"text": str(value)}] if value else []


def _iter_hf_contexts(
    dataset_name: str,
    split: str,
    cache_dir: Path | None,
    max_questions: int,
    seed: int,
) -> tuple[list[RawContext], str, int]:
    """Download and group a flattened Hugging Face KorQuAD split.

    ``LGCNS/KorQuAD_2.0`` exposes one row per question, while the rest of this
    script consumes SQuAD-style context objects. Rows are therefore grouped by
    title and exact context text before normal processing.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face source에는 datasets 패키지가 필요합니다. "
            "requirements.txt를 설치하세요."
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir.expanduser())
    try:
        rows = load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face 데이터셋을 내려받지 못했습니다: {dataset_name} ({split})"
        ) from exc

    if isinstance(rows, Mapping):
        raise ValueError(
            "Hugging Face split을 지정했지만 DatasetDict가 반환되었습니다: "
            f"{dataset_name} ({split})"
        )

    available_rows = len(rows)
    if max_questions > 0 and available_rows > max_questions:
        indices = random.Random(seed).sample(range(len(rows)), max_questions)
        rows = rows.select(indices)

    grouped: dict[tuple[str, str], RawContext] = {}
    source = f"hf://datasets/{dataset_name}@{split}"
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        raw_context = row.get("context")
        if not isinstance(raw_context, str) or not raw_context:
            continue
        context = raw_context
        title = str(row.get("title") or f"article_{row_index:06d}")
        key = (title, context)
        current = grouped.get(key)
        if current is None:
            current = RawContext(
                source,
                title,
                context,
                [],
                len(grouped),
            )
            grouped[key] = current
        question_id = str(row.get("id", "")).strip()
        if not question_id:
            question_id = f"hf_{row_index:08d}"
        answers = _hf_answer_list(row.get("answer", row.get("answers", [])))
        current.qas.append(
            {
                "id": question_id,
                "question": str(row.get("question") or ""),
                "answers": answers,
            }
        )
    contexts = list(grouped.values())
    if not contexts:
        raise ValueError(
            f"Hugging Face split에서 context를 찾지 못했습니다: {dataset_name} ({split})"
        )
    return contexts, source, available_rows


def _select_questions(
    questions: Sequence[QaInfo], max_questions: int, seed: int
) -> list[QaInfo]:
    """Select all questions or a reproducible random sample."""

    if max_questions == 0 or max_questions >= len(questions):
        return list(questions)
    return random.Random(seed).sample(list(questions), max_questions)


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in WORD_RE.finditer(text)]


def _locate_answer(
    context: str, answer: AnswerInfo, words: Sequence[tuple[int, int, str]]
) -> tuple[int, int] | None:
    start = answer.start
    if start is None or start < 0 or context[start : start + len(answer.text)] != answer.text:
        start = context.find(answer.text)
    if start < 0:
        return None
    end = start + len(answer.text)
    start_word = next(
        (index for index, (_, word_end, _) in enumerate(words) if word_end > start),
        None,
    )
    if start_word is None:
        return None
    end_word = next(
        (index for index, (word_start, _, _) in enumerate(words) if word_start >= end),
        len(words),
    )
    end_word = max(start_word + 1, end_word)
    return start_word, end_word


def _word_piece_counts(tokenizer: Any, words: Sequence[str]) -> list[int]:
    encoded = tokenizer(
        list(words),
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
    )
    word_ids = encoded.word_ids()
    if word_ids is None:
        return [
            max(1, len(tokenizer.encode(word, add_special_tokens=False)))
            for word in words
        ]
    counts = [0] * len(words)
    for word_id in word_ids:
        if word_id is not None and 0 <= word_id < len(counts):
            counts[word_id] += 1
    if any(count == 0 for count in counts):
        raise ValueError("tokenizer가 일부 어절을 token으로 변환하지 못했습니다.")
    return counts


def _partition_words(
    counts: Sequence[int],
    protected: Sequence[tuple[int, int]],
    budget: int,
) -> list[tuple[int, int]]:
    # ponytail: greedy word-boundary packing; replace only if utilization
    # measurements show that a more complex packing strategy is worthwhile.
    if budget < 1:
        raise ValueError("--max-context-tokens는 1 이상이어야 합니다.")
    windows: list[tuple[int, int]] = []
    total_words = len(counts)
    start = 0
    protected = sorted(set(protected))
    while start < total_words:
        end = start
        used = 0
        while end < total_words and used + counts[end] <= budget:
            used += counts[end]
            end += 1
        if end == start:
            raise ValueError(
                f"한 어절이 context token budget보다 깁니다: index={start}, "
                f"pieces={counts[start]}, budget={budget}"
            )

        while True:
            crossing = [
                interval
                for interval in protected
                if start <= interval[0] < end and interval[1] > end
            ]
            if not crossing:
                break
            first_start = min(interval[0] for interval in crossing)
            if first_start > start:
                end = first_start
                continue
            required_end = max(interval[1] for interval in crossing if interval[0] == start)
            required_pieces = sum(counts[start:required_end])
            if required_pieces > budget:
                raise ValueError(
                    "정답 span이 context token budget보다 깁니다: "
                    f"pieces={required_pieces}, budget={budget}"
                )
            end = required_end

        windows.append((start, end))
        start = end
    return windows


def _normalise_parser_words(parsed_words: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(word["text"]) for word in parsed_words)


def _span_record(
    sentence_id: str,
    parser: KoreanDependencyParser,
    text: str,
    max_span_length: int,
) -> tuple[str, dict[str, Any], int]:
    parsed = parser.parse_sentence(text)
    words = parsed["words"]
    if not words:
        raise ValueError(f"dependency parser가 빈 chunk를 반환했습니다: {sentence_id}")

    spans = generate_spans_up_to_L(words, max_span_length)
    output_spans: list[dict[str, Any]] = []
    for span in spans:
        zero_based = [int(word_id) - 1 for word_id in span["word_ids"]]
        begin, end = min(zero_based), max(zero_based) + 1
        output_spans.append(
            {
                "span_id": f"{sentence_id}_u0_{begin}_{end}",
                "text": span["text"],
                "eojeol_indices": zero_based,
                "size": len(zero_based),
                "is_contiguous": True,
                # The predictor batches records through the training collator,
                # which currently requires a label.  It is not a QA gold label.
                "label": "KEEP",
            }
        )

    record = {
        "sentence_id": sentence_id,
        "utterance_uid": f"{sentence_id}_u0",
        "utterance_start": 0,
        "utterance_end": len(words),
        "spans": output_spans,
    }
    return _normalise_parser_words(words), record, len(output_spans)


def _qa_payload(
    question: QaInfo,
) -> dict[str, Any]:
    return {
        "id": question.question_id,
        "question": question.question,
        "answers": [{"text": answer.text} for answer in question.answers],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="로컬 KorQuAD root directory, dev JSON, 또는 dev ZIP shard(s)",
    )
    source.add_argument(
        "--hf-dataset",
        default=None,
        help=(
            "Hugging Face dataset ID. 생략하면 LGCNS/KorQuAD_2.0의 "
            "validation split을 자동 다운로드"
        ),
    )
    parser.add_argument(
        "--hf-split",
        default="validation",
        help="Hugging Face split (KorQuAD dev에 해당하는 기본값: validation)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Hugging Face cache directory (선택)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="chunks.csv/qa_pairs.json/spans.jsonl 저장 위치",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="context window 계산에 사용할 tokenizer",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=3500,
        help="각 chunk의 tokenizer token 상한(질문/prompt 여유를 남김)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=1000,
        help="무작위로 선택할 QA 수; 0이면 dev 전체",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="QA 무작위 표본 추출 seed (재현성을 위해 고정)",
    )
    parser.add_argument(
        "--max-span-length",
        type=int,
        default=8,
        help="dependency span 후보의 최대 어절 수",
    )
    parser.add_argument(
        "--download-stanza",
        action="store_true",
        help="한국어 Stanza 모델을 먼저 다운로드",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="dependency parsing에 GPU 사용",
    )
    parser.add_argument(
        "--skip-unmatched",
        action="store_true",
        help="문맥에서 answer text를 찾지 못한 QA를 건너뜀",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_questions < 0:
        raise ValueError("--max-questions는 0 이상이어야 합니다.")
    if args.max_span_length < 1:
        raise ValueError("--max-span-length는 1 이상이어야 합니다.")

    source_paths: list[Path] = []
    source_names: list[str] = []
    available_source_questions: int | None = None
    if args.input:
        source_paths = _discover_inputs(args.input)
        contexts: list[RawContext] = []
        context_index = 0
        for source, payload in _iter_payloads(source_paths):
            source_names.append(source)
            for context in _iter_contexts(payload, source, context_index):
                contexts.append(context)
                context_index += 1
        source_type = "local"
        hf_dataset = None
        hf_split = None
    else:
        hf_dataset = args.hf_dataset or "LGCNS/KorQuAD_2.0"
        contexts, hf_source, available_source_questions = _iter_hf_contexts(
            hf_dataset,
            args.hf_split,
            args.hf_cache_dir,
            args.max_questions,
            args.seed,
        )
        source_names.append(hf_source)
        source_type = "huggingface"
        hf_split = args.hf_split
    if not contexts:
        raise ValueError("dev context를 찾지 못했습니다.")

    all_questions = _collect_questions(contexts)
    selected_questions = _select_questions(
        all_questions,
        args.max_questions,
        args.seed,
    )
    selected_id_order = [question.question_id for question in selected_questions]
    selected_ids = {question.question_id for question in selected_questions}
    selected_by_context: dict[int, list[QaInfo]] = {}
    for question in selected_questions:
        selected_by_context.setdefault(question.context_index, []).append(question)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("전처리에는 transformers가 필요합니다.") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("--tokenizer는 fast tokenizer여야 합니다.")

    if args.download_stanza:
        download_korean_model()
    dependency_parser = KoreanDependencyParser(use_gpu=args.gpu)

    chunks_rows: list[dict[str, str]] = []
    qa_passages: list[dict[str, Any]] = []
    span_records: list[dict[str, Any]] = []
    gold_contexts: list[dict[str, Any]] = []
    unmatched: list[str] = []
    chunk_count = 0
    span_count = 0
    max_observed_context_tokens = 0

    for context in contexts:
        questions = selected_by_context.get(context.context_index, [])
        if not questions:
            continue
        raw_words = _word_spans(context.context)
        words = [item[2] for item in raw_words]
        if not words:
            unmatched.extend(question.question_id for question in questions)
            continue
        counts = _word_piece_counts(tokenizer, words)
        protected: list[tuple[int, int]] = []
        for question in questions:
            interval = None
            for answer in question.answers:
                interval = _locate_answer(context.context, answer, raw_words)
                if interval is not None:
                    break
            if interval is None:
                unmatched.append(question.question_id)
                continue
            question.word_interval = interval
            protected.append(interval)

        usable_questions = [
            question
            for question in questions
            if question.question_id not in unmatched
        ]
        if not usable_questions:
            continue
        windows = _partition_words(
            counts,
            [question.word_interval for question in usable_questions if question.word_interval],
            args.max_context_tokens,
        )
        context_gold_qas: list[dict[str, Any]] = []
        for question in usable_questions:
            context_gold_qas.append(question.raw)

        gold_contexts.append(
            {
                "title": context.title,
                "context": context.context,
                "qas": context_gold_qas,
            }
        )

        for window_index, (begin, end) in enumerate(windows):
            window_questions = [
                question
                for question in usable_questions
                if question.word_interval
                and begin <= question.word_interval[0]
                and question.word_interval[1] <= end
            ]
            if not window_questions:
                continue
            sentence_id = f"korquad_{context.context_index:06d}_w{window_index:04d}"
            raw_window_text = " ".join(words[begin:end])
            sentence, record, record_span_count = _span_record(
                sentence_id,
                dependency_parser,
                raw_window_text,
                args.max_span_length,
            )
            observed_tokens = len(
                tokenizer(sentence, add_special_tokens=False)["input_ids"]
            )
            if observed_tokens > args.max_context_tokens:
                raise ValueError(
                    "dependency tokenization 후 chunk가 context budget을 초과했습니다: "
                    f"{sentence_id} ({observed_tokens} > {args.max_context_tokens})"
                )
            max_observed_context_tokens = max(
                max_observed_context_tokens, observed_tokens
            )
            chunks_rows.append({"sentence_id": sentence_id, "sentence": sentence})
            qa_passages.append(
                {
                    "passage_id": sentence_id,
                    "context": sentence,
                    "qas": [_qa_payload(question) for question in window_questions],
                }
            )
            span_records.append(record)
            chunk_count += 1
            span_count += record_span_count

    if unmatched and not args.skip_unmatched:
        raise ValueError(
            "answer text를 문맥에서 찾지 못한 QA가 있습니다: "
            f"{unmatched[:10]} (총 {len(unmatched)}개). "
            "필요하면 --skip-unmatched를 사용하세요."
        )
    if args.skip_unmatched and unmatched:
        unmatched_set = set(unmatched)
        for passage in qa_passages:
            passage["qas"] = [
                qa for qa in passage["qas"] if qa["id"] not in unmatched_set
            ]
        qa_passages = [passage for passage in qa_passages if passage["qas"]]
        selected_ids -= unmatched_set
        selected_id_order = [
            question_id
            for question_id in selected_id_order
            if question_id not in unmatched_set
        ]

    if not qa_passages:
        raise ValueError("QA가 포함된 chunk를 만들지 못했습니다.")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["sentence_id", "sentence"])
        writer.writeheader()
        writer.writerows(chunks_rows)
    _write_json(output_dir / "qa_pairs.json", qa_passages)
    with (output_dir / "spans.jsonl").open("w", encoding="utf-8") as handle:
        for record in span_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write_json(
        output_dir / "gold_subset.json",
        {
            "version": "korpress-korquad-dev-subset",
            "data": [
                {
                    "title": context["title"],
                    "paragraphs": [
                        {
                            "context": context["context"],
                            "qas": context["qas"],
                        }
                    ],
                }
                for context in gold_contexts
            ],
        },
    )
    _write_json(output_dir / "question_ids.json", selected_id_order)
    manifest = {
        "source_type": source_type,
        "source_paths": [str(path) for path in source_paths],
        "dev_sources": source_names,
        "hf_dataset": hf_dataset,
        "hf_split": hf_split,
        "tokenizer": args.tokenizer,
        "max_context_tokens": args.max_context_tokens,
        "max_observed_context_tokens": max_observed_context_tokens,
        "max_questions_requested": args.max_questions,
        "question_selection": (
            "all" if args.max_questions == 0 else "random_sample"
        ),
        "seed": args.seed,
        "questions_available": (
            available_source_questions
            if available_source_questions is not None
            else len(all_questions)
        ),
        "questions_loaded_before_selection": len(all_questions),
        "questions_selected": len(selected_ids),
        "contexts_with_selected_questions": len(gold_contexts),
        "chunks": chunk_count,
        "span_records": len(span_records),
        "spans": span_count,
        "unmatched_question_ids": unmatched,
        "max_span_length": args.max_span_length,
        "span_label_placeholder": "KEEP",
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
