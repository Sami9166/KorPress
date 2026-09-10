"""Measure official KorQuAD article lengths with the experiment tokenizers.

Accepts extracted JSON files or the official dev ZIP shards.  This is a
read-only inspection command; it does not run compression or QA generation.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterator, Mapping


def _payloads(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".json"):
                    continue
                with archive.open(name) as handle:
                    payload = json.load(io.TextIOWrapper(handle, encoding="utf-8"))
                if isinstance(payload, Mapping):
                    yield f"{path.name}:{name}", payload
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"KorQuAD JSON 최상위 객체가 아닙니다: {path}")
    yield str(path), payload


def _contexts(payload: Mapping[str, Any]) -> Iterator[tuple[str, str, int, int]]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("KorQuAD JSON의 data가 배열이 아닙니다.")
    for article_index, article in enumerate(data):
        if not isinstance(article, Mapping):
            continue
        # KorQuAD 2.1 dev shards store one QA context directly on each
        # article.  The older SQuAD-shaped export stores contexts in
        # ``paragraphs``; support both without merging unrelated contexts.
        if "context" in article:
            qas = article.get("qas", [])
            yield (
                str(article.get("title", f"article_{article_index:06d}")),
                str(article.get("context", "")),
                len(qas) if isinstance(qas, list) else 0,
                0,
            )
            continue
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            if not isinstance(paragraph, Mapping):
                continue
            context = str(paragraph.get("context", ""))
            qas = paragraph.get("qas", [])
            qa_count = len(qas) if isinstance(qas, list) else 0
            yield (
                str(article.get("title", f"article_{article_index:06d}")),
                context,
                qa_count,
                paragraph_index,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=("Qwen/Qwen3-8B", "klue/roberta-base"),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizers = {
        name: AutoTokenizer.from_pretrained(name, use_fast=True)
        for name in args.tokenizers
    }
    maxima: dict[str, dict[str, Any]] = {
        name: {"tokens": -1} for name in tokenizers
    }
    article_count = 0
    context_count = 0
    qa_count = 0
    for path in args.input:
        for source, payload in _payloads(path):
            article_count += len(payload.get("data", []))
            for title, context, count, paragraph_index in _contexts(payload):
                context_count += 1
                qa_count += count
                for name, tokenizer in tokenizers.items():
                    token_count = len(
                        tokenizer(context, add_special_tokens=False)["input_ids"]
                    )
                    if token_count > maxima[name]["tokens"]:
                        maxima[name] = {
                            "tokens": token_count,
                            "characters": len(context),
                            "whitespace_words": len(context.split()),
                            "title": title,
                            "paragraph_index": paragraph_index,
                            "source": source,
                        }

    result = {
        "articles": article_count,
        "contexts": context_count,
        "qa_pairs": qa_count,
        "max_context_by_tokenizer": maxima,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
