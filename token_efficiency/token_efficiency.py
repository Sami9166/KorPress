"""Core data, compression, API, evidence, and metric logic for token efficiency."""

from __future__ import annotations

import difflib
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

LANGS = ("en", "ko")
BELEBELE_CONFIGS = {"en": "eng_Latn", "ko": "kor_Hang"}
RATES = (0.33, 0.50, 0.75)


# ---------------------------------------------------------------------------
# Belebele data

def item_key(row: Mapping[str, Any]) -> str:
    return f"{row['link']}::{row['question_number']}"


def load_belebele() -> dict[str, dict[str, dict[str, Any]]]:
    """Load both language splits and validate their parallel keys."""

    from datasets import load_dataset

    data: dict[str, dict[str, dict[str, Any]]] = {}
    for lang in LANGS:
        rows = load_dataset("facebook/belebele", BELEBELE_CONFIGS[lang])["test"]
        data[lang] = {item_key(row): dict(row) for row in rows}
    validate_parallel_keys(data)
    return data


def validate_parallel_keys(data: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    if not all(lang in data for lang in LANGS):
        raise ValueError(f"Expected languages {LANGS}, got {tuple(data)}")
    key_sets = [set(data[lang]) for lang in LANGS]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("English and Korean Belebele key sets do not match")
    mismatches = [
        key
        for key in key_sets[0]
        if str(data["en"][key]["correct_answer_num"])
        != str(data["ko"][key]["correct_answer_num"])
    ]
    if mismatches:
        raise ValueError(f"Answer-number mismatch for {len(mismatches)} parallel items")


def select_stratified_sample(
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    n: int = 300,
    seed: int = 20260904,
) -> list[str]:
    """Select one question per passage and 25% per answer position."""

    if n <= 0 or n % 4:
        raise ValueError("n must be a positive multiple of four")
    validate_parallel_keys(data)
    by_passage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in data["en"].values():
        by_passage[str(row["link"])].append(row)

    rng = random.Random(seed)
    passages = list(by_passage)
    rng.shuffle(passages)
    candidates: dict[str, dict[str, Mapping[str, Any]]] = {}
    for passage, rows in by_passage.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["correct_answer_num"])].append(row)
        candidates[passage] = {
            answer: rng.choice(items) for answer, items in grouped.items()
        }

    quota = {str(i): n // 4 for i in range(1, 5)}
    chosen: list[Mapping[str, Any]] = []
    used_passages: set[str] = set()
    answer_order = sorted(
        quota,
        key=lambda answer: sum(answer in candidates[p] for p in passages),
    )
    for answer in answer_order:
        for passage in passages:
            if quota[answer] == 0:
                break
            if passage in used_passages or answer not in candidates[passage]:
                continue
            chosen.append(candidates[passage][answer])
            used_passages.add(passage)
            quota[answer] -= 1
        if quota[answer]:
            raise ValueError(f"Could not fill answer-position quota for {answer}")

    rng.shuffle(chosen)
    keys = [item_key(row) for row in chosen]
    counts = Counter(str(data["en"][key]["correct_answer_num"]) for key in keys)
    if len(keys) != n or any(counts[str(i)] != n // 4 for i in range(1, 5)):
        raise AssertionError(f"Unexpected sample distribution: {counts}")
    return keys


def save_sample_ids(keys: Sequence[str], path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"n": len(keys), "seed": seed, "keys": list(keys)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_sample_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = payload.get("keys")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError(f"Invalid sample-id file: {path}")
    return keys


# ---------------------------------------------------------------------------
# LLMLingua-2 compression and token accounting

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


class LLMLingua2Compressor:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        import torch
        from llmlingua import PromptCompressor

        device_map = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map,
        )

        import tiktoken

        self.gpt_encoder = tiktoken.get_encoding("o200k_base")
        self.xlmr_tokenizer = self.model.tokenizer

    def compress(self, text: str, rate: float) -> str:
        result = self.model.compress_prompt(
            text,
            rate=rate,
            force_tokens=[],
            drop_consecutive=False,
        )
        return str(result["compressed_prompt"])

    def gpt_tokens(self, text: str) -> int:
        return len(self.gpt_encoder.encode(text))

    def xlmr_tokens(self, text: str) -> int:
        return len(self.xlmr_tokenizer.encode(text, add_special_tokens=False))


def _successful_compressions(path: Path) -> set[tuple[str, str, float]]:
    return {
        (str(record["key"]), str(record["lang"]), float(record["rate"]))
        for record in load_jsonl(path)
        if record.get("error") is None and {"key", "lang", "rate"}.issubset(record)
    }


def compress_dataset(
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    keys: Iterable[str],
    rates: Iterable[float],
    output_path: Path,
    model_name: str,
    langs: Iterable[str] = LANGS,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run all language/rate combinations and append resumable JSONL records."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = LLMLingua2Compressor(model_name, device=device)
    keys, rates, langs = list(keys), tuple(map(float, rates)), tuple(langs)
    jobs = [(key, lang, rate) for lang in langs for rate in rates for key in keys]
    done = _successful_compressions(output_path)
    todo = [job for job in jobs if job not in done]
    print(f"Compression jobs: {len(jobs)} total, {len(todo)} remaining")

    started = time.time()
    with output_path.open("a", encoding="utf-8") as handle:
        for index, (key, lang, rate) in enumerate(todo, start=1):
            original = str(data[lang][key]["flores_passage"])
            record: dict[str, Any] = {"key": key, "lang": lang, "rate": rate}
            try:
                compressed = compressor.compress(original, rate)
                record.update(
                    compressed=compressed,
                    error=None,
                    xlmr_before=compressor.xlmr_tokens(original),
                    xlmr_after=compressor.xlmr_tokens(compressed),
                    gpt_before=compressor.gpt_tokens(original),
                    gpt_after=compressor.gpt_tokens(compressed),
                    chunked=int("\n" in compressed),
                    empty=int(not compressed.strip()),
                )
            except Exception as exc:  # preserve the rest of a long run
                record["error"] = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 100 == 0:
                print(f"  {index}/{len(todo)} ({time.time() - started:.0f}s)")
    return load_jsonl(output_path)


# ---------------------------------------------------------------------------
# QA and evidence prompts

SYSTEM = (
    "You are answering a multiple-choice question. "
    "Your entire response must be exactly one character: 1, 2, 3, or 4. "
    "Do not write any words, explanation, or punctuation. "
    "If no passage is given or information is missing, still output your best "
    "guess as a single digit. Never refuse."
)

EXTRACT_SYSTEM = (
    "You extract evidence spans from a passage. "
    "Given a passage, a question, and the correct answer, quote the minimal "
    "part(s) of the passage that justify that answer. "
    "Copy the text VERBATIM from the passage - do not translate, paraphrase, "
    "reorder, or fix anything. Keep quotes short and contiguous. "
    'Respond with JSON only: {"quotes": ["...", "..."]}. No other text.'
)


def build_prompt(row: Mapping[str, Any], passage: str | None = None) -> str:
    body = "" if passage is None else f"Passage:\n{passage}\n\n"
    return (
        body
        + f"Question: {row['question']}\n"
        + f"1. {row['mc_answer1']}\n2. {row['mc_answer2']}\n"
        + f"3. {row['mc_answer3']}\n4. {row['mc_answer4']}\n\n"
        + "Answer with a single digit (1-4):"
    )


def parse_answer(text: str | None) -> int | None:
    if not text:
        return None
    allowed_prefix = "\"'`*.-(["
    for char in text.strip():
        if char in "1234":
            return int(char)
        if not char.isspace() and char not in allowed_prefix:
            return None
    return None


def extract_prompt(row: Mapping[str, Any]) -> str:
    answer = row[f"mc_answer{int(row['correct_answer_num'])}"]
    return (
        f"Passage:\n{row['flores_passage']}\n\n"
        f"Question: {row['question']}\nCorrect answer: {answer}\n\n"
        "Quote the minimal verbatim span(s) from the passage that justify the correct answer."
    )


def parse_quotes(text: str | None) -> list[str] | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    quotes = payload.get("quotes")
    return (
        [quote for quote in quotes if isinstance(quote, str) and quote.strip()]
        if isinstance(quotes, list)
        else None
    )


def build_evidence_jobs(
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {"key": key, "lang": lang, "cond": "evidence", "user_text": extract_prompt(data[lang][key])}
        for lang in LANGS
        for key in keys
    ]


def build_eval_jobs(
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    keys: Sequence[str],
    compressed: Sequence[Mapping[str, Any]],
    rates: Iterable[float] = RATES,
) -> list[dict[str, Any]]:
    rates = tuple(map(float, rates))
    by_key = {
        (record["key"], record["lang"], float(record["rate"])): record
        for record in compressed
        if record.get("error") is None
    }
    jobs: list[dict[str, Any]] = []
    for lang in LANGS:
        for key in keys:
            row = data[lang][key]
            base = {"key": key, "lang": lang, "gold": int(row["correct_answer_num"])}
            jobs.append({**base, "cond": "blind", "rate": None, "user_text": build_prompt(row)})
            jobs.append(
                {
                    **base,
                    "cond": "full",
                    "rate": None,
                    "user_text": build_prompt(row, row["flores_passage"]),
                }
            )
            for rate in rates:
                record = by_key.get((key, lang, rate))
                if record is None:
                    raise KeyError(f"Missing compressed record for {(key, lang, rate)}")
                jobs.append(
                    {
                        **base,
                        "cond": f"r{rate:g}",
                        "rate": rate,
                        "gpt_after": record.get("gpt_after"),
                        "user_text": build_prompt(row, record["compressed"]),
                    }
                )
    return jobs


def score_answer(record: Mapping[str, Any], text: str | None) -> dict[str, Any]:
    prediction = parse_answer(text)
    return {
        "pred": prediction,
        "correct": None if prediction is None else int(prediction == record["gold"]),
        "parse_fail": int(prediction is None and record.get("error") is None),
        "raw": (text or "")[:60],
    }


# ---------------------------------------------------------------------------
# Anthropic runner

class AnthropicRunner:
    RETRYABLE_ERRORS = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "APIStatusError",
    }

    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.client = anthropic.Anthropic(api_key=key)

    def call(
        self,
        text: str,
        system: str,
        max_tokens: int,
        retries: int = 6,
    ) -> tuple[str | None, Any, str | None]:
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": text}],
                    extra_body={"temperature": 0},
                )
                if getattr(response, "stop_reason", None) == "refusal":
                    return None, getattr(response, "usage", None), "refusal"
                content = getattr(response, "content", ())
                output = "".join(
                    block.text for block in content if getattr(block, "type", None) == "text"
                )
                return output, getattr(response, "usage", None), None
            except Exception as exc:
                name = type(exc).__name__
                if name not in self.RETRYABLE_ERRORS or attempt == retries - 1:
                    return None, None, f"{name}: {str(exc)[:150]}"
                time.sleep(min(2**attempt, 30) * (1 + random.random() * 0.3))
        return None, None, "exhausted"


def _completed_api_jobs(path: Path) -> set[tuple[Any, Any, Any]]:
    return {
        (record["key"], record["lang"], record["cond"])
        for record in load_jsonl(path)
        if record.get("error") is None and {"key", "lang", "cond"}.issubset(record)
    }


def run_jobs(
    jobs: Sequence[Mapping[str, Any]],
    output_path: Path,
    runner: AnthropicRunner,
    postprocess: Callable[[Mapping[str, Any], str | None], Mapping[str, Any]],
    system: str,
    max_tokens: int,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Run API jobs concurrently; successful JSONL records are checkpointed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_api_jobs(output_path)
    todo = [
        dict(job)
        for job in jobs
        if (job.get("key"), job.get("lang"), job.get("cond")) not in done
    ]
    print(f"API jobs: {len(jobs)} total, {len(todo)} remaining")
    if not todo:
        return load_jsonl(output_path)

    lock = threading.Lock()
    with output_path.open("a", encoding="utf-8") as handle:
        completed = [0]

        def work(job: dict[str, Any]) -> None:
            user_text = str(job.pop("user_text"))
            text, usage, error = runner.call(user_text, system, max_tokens)
            record = {
                **job,
                "error": error,
                "in_tok": getattr(usage, "input_tokens", None) if usage else None,
                "out_tok": getattr(usage, "output_tokens", None) if usage else None,
            }
            record.update(postprocess(record, text))
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[0] += 1
                if completed[0] % 250 == 0:
                    print(f"  {completed[0]}/{len(todo)}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(work, todo))
    return load_jsonl(output_path)


# ---------------------------------------------------------------------------
# Evidence alignment

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_word(word: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFC", word)).lower()


def snap_to_words(quote: str, passage: str) -> set[int] | None:
    normalized_words: list[str] = []
    owner: list[int] = []
    for index, word in enumerate(passage.split()):
        normalized = normalize_word(word)
        normalized_words.append(normalized)
        owner.extend([index] * len(normalized))
    passage_compact = "".join(normalized_words)
    quote_compact = "".join(normalize_word(word) for word in quote.split())
    if not passage_compact or not quote_compact:
        return None
    position = passage_compact.find(quote_compact)
    return set(owner[position : position + len(quote_compact)]) if position >= 0 else None


def build_evidence_map(
    records: Sequence[Mapping[str, Any]],
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[tuple[str, str], tuple[list[str], list[int]]]:
    evidence: dict[tuple[str, str], tuple[list[str], list[int]]] = {}
    for record in records:
        if record.get("error") or not record.get("quotes"):
            continue
        lang, key = str(record["lang"]), str(record["key"])
        passage = str(data[lang][key]["flores_passage"])
        quotes = [quote for quote in record["quotes"] if isinstance(quote, str)]
        indices: set[int] = set()
        for quote in quotes:
            snapped = snap_to_words(quote, passage)
            if snapped:
                indices.update(snapped)
        if indices:
            evidence[(lang, key)] = (quotes, sorted(indices))
    return evidence


def kept_mask(original: str, compressed: str) -> list[bool]:
    original_words, compressed_words = original.split(), compressed.split()
    matcher = difflib.SequenceMatcher(
        None,
        [normalize_word(word) for word in original_words],
        [normalize_word(word) for word in compressed_words],
        autojunk=False,
    )
    mask = [False] * len(original_words)
    for tag, start, end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            for index in range(start, end):
                mask[index] = True
    return mask


def evidence_survival_records(
    compressed: Sequence[Mapping[str, Any]],
    evidence: Mapping[tuple[str, str], tuple[list[str], list[int]]],
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in compressed:
        if record.get("error") is not None:
            continue
        key = (str(record["lang"]), str(record["key"]))
        if key not in evidence:
            continue
        original = str(data[key[0]][key[1]]["flores_passage"])
        mask = kept_mask(original, str(record["compressed"]))
        indices = [index for index in evidence[key][1] if index < len(mask)]
        if not indices:
            continue
        records.append(
            {
                "lang": key[0],
                "key": key[1],
                "rate": float(record["rate"]),
                "ev_survival": sum(mask[index] for index in indices) / len(indices),
                "ev_kept": sum(mask[index] for index in indices),
                "ev_total": len(indices),
                "gpt_kept": float(record["gpt_after"]) / float(record["gpt_before"]),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Metrics and CSV export

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - half_width, centre + half_width


def token_table(records):
    import pandas as pd

    frame = records.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "lang", "rate", "gpt_after", "gpt_before", "xlmr_after", "xlmr_before",
                "n", "chunked", "empty", "gpt_kept", "xlmr_kept",
            ]
        )
    if "error" in frame:
        frame = frame[frame["error"].isna()]
    table = (
        frame.groupby(["lang", "rate"])
        .agg(
            gpt_after=("gpt_after", "mean"),
            gpt_before=("gpt_before", "mean"),
            xlmr_after=("xlmr_after", "mean"),
            xlmr_before=("xlmr_before", "mean"),
            n=("key", "size"),
            chunked=("chunked", "sum"),
            empty=("empty", "sum"),
        )
        .reset_index()
    )
    table["gpt_kept"] = table["gpt_after"] / table["gpt_before"]
    table["xlmr_kept"] = table["xlmr_after"] / table["xlmr_before"]
    return table


def accuracy_table(records, subset: str = "all"):
    import pandas as pd

    frame = records.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["subset", "lang", "cond", "n", "n_parse_fail", "acc", "ci_lo", "ci_hi"]
        )
    if "error" in frame:
        frame = frame[frame["error"].isna()]
    if "parse_fail" not in frame:
        frame["parse_fail"] = 0
    rows: list[dict[str, Any]] = []
    for (lang, condition), group in frame.groupby(["lang", "cond"], dropna=False):
        scored = group[group["correct"].notna()]
        successes, total = int(scored["correct"].sum()), len(scored)
        low, high = wilson_interval(successes, total)
        rows.append(
            {
                "subset": subset,
                "lang": lang,
                "cond": condition,
                "n": total,
                "n_parse_fail": int(group["parse_fail"].fillna(0).sum()),
                "acc": successes / total if total else float("nan"),
                "ci_lo": low,
                "ci_hi": high,
            }
        )
    return pd.DataFrame(rows).sort_values(["lang", "cond"]).reset_index(drop=True)


def passage_dependent_records(records):
    import pandas as pd

    if records.empty:
        return records.copy()
    blind = records[records["cond"] == "blind"]
    hard_keys = {(row.lang, row.key) for row in blind.itertuples() if row.correct == 0}
    pairs = pd.MultiIndex.from_frame(records[["lang", "key"]])
    return records[pairs.isin(hard_keys)].copy()


def transfer_efficiency(accuracy, rates: Iterable[float] = RATES):
    import pandas as pd

    values = accuracy.set_index(["lang", "cond"])["acc"]
    rows: list[dict[str, Any]] = []
    for lang in sorted(accuracy["lang"].unique()):
        denominator = float(values[(lang, "full")] - values[(lang, "blind")])
        for rate in rates:
            condition = f"r{float(rate):g}"
            efficiency = (
                float(values[(lang, condition)] - values[(lang, "blind")]) / denominator
                if denominator
                else float("nan")
            )
            rows.append(
                {
                    "lang": lang,
                    "cond": condition,
                    "denom": denominator,
                    "eff": efficiency,
                    "unstable": int(denominator < 0.15),
                }
            )
    return pd.DataFrame(rows)


def evidence_survival_table(records):
    import pandas as pd

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["lang", "rate", "n", "ev_survival", "gpt_kept", "ev_minus_token"]
        )
    table = (
        frame.groupby(["lang", "rate"])
        .agg(n=("ev_survival", "size"), ev_survival=("ev_survival", "mean"), gpt_kept=("gpt_kept", "mean"))
        .reset_index()
    )
    table["ev_minus_token"] = table["ev_survival"] - table["gpt_kept"]
    return table


def matched_budget_table(token_metrics, accuracy, hard_accuracy, efficiency):
    import pandas as pd

    if token_metrics.empty or hard_accuracy.empty:
        return pd.DataFrame()
    acc = accuracy.set_index(["lang", "cond"])["acc"]
    hard = hard_accuracy.set_index(["lang", "cond"])["acc"]
    eff = efficiency.set_index(["lang", "cond"])["eff"]
    rows: list[dict[str, Any]] = []
    for lang in sorted(token_metrics["lang"].unique()):
        subset = token_metrics[token_metrics["lang"] == lang]
        for budget in (45.0, 71.0):
            nearest = subset.loc[(subset["gpt_after"] - budget).abs().idxmin()]
            condition = f"r{float(nearest['rate']):g}"
            rows.append(
                {
                    "budget": f"~{budget:g} tok",
                    "lang": lang,
                    "rate": nearest["rate"],
                    "gpt_after": nearest["gpt_after"],
                    "acc": acc[(lang, condition)],
                    "acc_hard": hard[(lang, condition)],
                    "eff": eff[(lang, condition)],
                }
            )
    return pd.DataFrame(rows)


def save_csv_tables(tables: Mapping[str, Any], output_dir: Path) -> None:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        if len(pd.read_csv(path)) != len(table):
            raise AssertionError(f"CSV row-count mismatch for {path}")
        print(f"{path.name}: {len(table)} rows")


def read_jsonl_frame(path: Path):
    import pandas as pd

    return pd.DataFrame(load_jsonl(path))


# ---------------------------------------------------------------------------
# Optional figures

def plot_tradeoff(token_metrics, accuracy, survival, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = token_metrics.set_index(["lang", "rate"])
    scores = accuracy.set_index(["lang", "cond"])
    evidence = survival.set_index(["lang", "rate"])
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5))
    styles = {"en": ("#1f77b4", "o", "English"), "ko": ("#d62728", "s", "Korean")}
    for lang, (color, marker, label) in styles.items():
        rates = sorted(token_metrics[token_metrics.lang == lang].rate.unique())
        xs = [tokens.loc[(lang, rate), "gpt_after"] for rate in rates]
        ys = [scores.loc[(lang, f"r{rate:g}"), "acc"] for rate in rates]
        axes[0].plot(xs, ys, color=color, marker=marker, lw=1.9, label=label)
        axes[0].axhline(scores.loc[(lang, "blind"), "acc"], ls=":", color=color, alpha=0.7)
        axes[1].plot(
            xs,
            [evidence.loc[(lang, rate), "ev_survival"] for rate in rates],
            color=color,
            marker=marker,
            lw=1.9,
            label=label,
        )
        axes[1].plot(
            xs,
            [evidence.loc[(lang, rate), "gpt_kept"] for rate in rates],
            color=color,
            ls="--",
            alpha=0.55,
            label=f"{label} all tokens",
        )
    axes[0].set(xlabel="Passage tokens after compression (o200k_base)", ylabel="Accuracy", title="Accuracy vs token budget")
    axes[1].set(xlabel="Passage tokens after compression (o200k_base)", ylabel="Answer-evidence survival", title="Evidence survival vs token budget")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_passage_dependent(token_metrics, hard_accuracy, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = token_metrics.set_index(["lang", "rate"])
    scores = hard_accuracy.set_index(["lang", "cond"])
    figure, axis = plt.subplots(figsize=(7, 5))
    for lang, color, marker, label in (("en", "#1f77b4", "o", "English"), ("ko", "#d62728", "s", "Korean")):
        rates = sorted(token_metrics[token_metrics.lang == lang].rate.unique())
        axis.plot(
            [tokens.loc[(lang, rate), "gpt_after"] for rate in rates],
            [scores.loc[(lang, f"r{rate:g}"), "acc"] for rate in rates],
            color=color,
            marker=marker,
            lw=1.9,
            label=label,
        )
    axis.set(xlabel="Passage tokens after compression (o200k_base)", ylabel="Accuracy (passage-dependent items)", title="Passage-dependent accuracy")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
