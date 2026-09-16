"""Run the Belebele token-efficiency experiment.

The pipeline is resumable: compression and API commands append JSONL
checkpoints, while ``metrics`` turns them into the CSV tables in ``results``.
Run from the repository root, for example::

    python token_efficiency/run_token_efficiency.py self-check
    python token_efficiency/run_token_efficiency.py sample
    python token_efficiency/run_token_efficiency.py compress
    python token_efficiency/run_token_efficiency.py evidence
    python token_efficiency/run_token_efficiency.py evaluate
    python token_efficiency/run_token_efficiency.py metrics
    python token_efficiency/run_token_efficiency.py plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import token_efficiency as core  # noqa: E402


DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figs"
DEFAULT_COMPRESSOR = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
DEFAULT_EVAL_MODEL = "claude-haiku-4-5-20251001"


def path_arg(value: str | None, default: Path) -> Path:
    path = default if value is None else Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sample(args: argparse.Namespace) -> None:
    data = core.load_belebele()
    keys = core.select_stratified_sample(data, n=args.n, seed=args.seed)
    output = path_arg(args.output, DATA_DIR / "sample_ids.json")
    core.save_sample_ids(keys, output, seed=args.seed)
    print(f"Saved {len(keys)} sample IDs to {output}")


def compress(args: argparse.Namespace) -> None:
    data = core.load_belebele()
    sample_path = path_arg(args.sample, DATA_DIR / "sample_ids.json")
    output = path_arg(args.output, DATA_DIR / "compressed.jsonl")
    records = core.compress_dataset(
        data,
        core.load_sample_ids(sample_path),
        core.RATES,
        output_path=output,
        model_name=args.model,
        device=args.device,
    )
    successful = sum(record.get("error") is None for record in records)
    print(f"Saved {successful}/{len(records)} successful compression records to {output}")


def evidence(args: argparse.Namespace) -> None:
    data = core.load_belebele()
    keys = core.load_sample_ids(path_arg(args.sample, DATA_DIR / "sample_ids.json"))
    output = path_arg(args.output, DATA_DIR / "evidence.jsonl")
    runner = core.AnthropicRunner(args.model)
    core.run_jobs(
        core.build_evidence_jobs(data, keys),
        output_path=output,
        runner=runner,
        postprocess=lambda _record, text: {"quotes": core.parse_quotes(text)},
        system=core.EXTRACT_SYSTEM,
        max_tokens=400,
        workers=args.workers,
    )
    print(f"Saved evidence responses to {output}")


def evaluate(args: argparse.Namespace) -> None:
    data = core.load_belebele()
    keys = core.load_sample_ids(path_arg(args.sample, DATA_DIR / "sample_ids.json"))
    compressed = core.load_jsonl(path_arg(args.compression, DATA_DIR / "compressed.jsonl"))
    output = path_arg(args.output, DATA_DIR / "eval.jsonl")
    runner = core.AnthropicRunner(args.model)
    core.run_jobs(
        core.build_eval_jobs(data, keys, compressed, rates=core.RATES),
        output_path=output,
        runner=runner,
        postprocess=core.score_answer,
        system=core.SYSTEM,
        max_tokens=16,
        workers=args.workers,
    )
    print(f"Saved evaluation responses to {output}")


def metrics(args: argparse.Namespace) -> None:
    import pandas as pd

    data = core.load_belebele()
    compressed = core.load_jsonl(path_arg(args.compression, DATA_DIR / "compressed.jsonl"))
    evaluation = core.read_jsonl_frame(path_arg(args.evaluation, DATA_DIR / "eval.jsonl"))
    all_accuracy = core.accuracy_table(evaluation, subset="all")
    hard_accuracy = core.accuracy_table(
        core.passage_dependent_records(evaluation), subset="passage-dependent"
    )
    token_metrics = core.token_table(pd.DataFrame(compressed))
    efficiency = core.transfer_efficiency(all_accuracy)

    evidence_records = core.load_jsonl(path_arg(args.evidence, DATA_DIR / "evidence.jsonl"))
    evidence_map = core.build_evidence_map(evidence_records, data)
    survival_records = core.evidence_survival_records(compressed, evidence_map, data)
    survival = core.evidence_survival_table(survival_records)
    tables = {
        "token_metrics": token_metrics,
        "accuracy": all_accuracy,
        "accuracy_hard": hard_accuracy,
        "transfer_efficiency": efficiency,
        "evidence_survival": survival,
        "evidence_per_item": pd.DataFrame(
            survival_records
        ),
    }
    if not token_metrics.empty and not hard_accuracy.empty:
        tables["matched_budget"] = core.matched_budget_table(
            token_metrics, all_accuracy, hard_accuracy, efficiency
        )
    core.save_csv_tables(tables, path_arg(args.output_dir, RESULTS_DIR))


def plot(args: argparse.Namespace) -> None:
    import pandas as pd

    result_dir = path_arg(args.results, RESULTS_DIR)
    output_dir = path_arg(args.output_dir, FIGURES_DIR)
    core.plot_tradeoff(
        pd.read_csv(result_dir / "token_metrics.csv"),
        pd.read_csv(result_dir / "accuracy.csv"),
        pd.read_csv(result_dir / "evidence_survival.csv"),
        output_dir / "accuracy_evidence_tradeoff.png",
    )
    core.plot_passage_dependent(
        pd.read_csv(result_dir / "token_metrics.csv"),
        pd.read_csv(result_dir / "accuracy_hard.csv"),
        output_dir / "passage_dependent_accuracy.png",
    )
    print(f"Saved figures to {output_dir}")


def self_check(_args: argparse.Namespace) -> None:
    assert core.parse_answer(" **2** ") == 2
    assert core.parse_answer("answer: 2") is None
    assert core.parse_quotes('{"quotes": ["가 나", ""]}') == ["가 나"]
    assert core.snap_to_words("바나나", "사과 바나나") == {1}
    assert core.kept_mask("a b c", "a c") == [True, False, True]
    assert 0 < core.wilson_interval(1, 1)[0] < 1
    print("SELF_CHECK_OK")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("sample", help="download Belebele and select a stratified sample")
    command.add_argument("--n", type=int, default=300)
    command.add_argument("--seed", type=int, default=20260904)
    command.add_argument("--output")
    command.set_defaults(func=sample)

    command = commands.add_parser("compress", help="compress sampled passages with LLMLingua-2")
    command.add_argument("--sample")
    command.add_argument("--output")
    command.add_argument("--model", default=DEFAULT_COMPRESSOR)
    command.add_argument("--device", choices=("cpu", "cuda"))
    command.set_defaults(func=compress)

    command = commands.add_parser("evidence", help="extract answer-evidence quotes via Anthropic")
    command.add_argument("--sample")
    command.add_argument("--output")
    command.add_argument("--model", default=DEFAULT_EVAL_MODEL)
    command.add_argument("--workers", type=int, default=8)
    command.set_defaults(func=evidence)

    command = commands.add_parser("evaluate", help="run blind/full/compressed QA calls")
    command.add_argument("--sample")
    command.add_argument("--compression")
    command.add_argument("--output")
    command.add_argument("--model", default=DEFAULT_EVAL_MODEL)
    command.add_argument("--workers", type=int, default=8)
    command.set_defaults(func=evaluate)

    command = commands.add_parser("metrics", help="aggregate JSONL outputs into result CSVs")
    command.add_argument("--compression")
    command.add_argument("--evaluation")
    command.add_argument("--evidence")
    command.add_argument("--output-dir")
    command.set_defaults(func=metrics)

    command = commands.add_parser("plot", help="render the result figures")
    command.add_argument("--results")
    command.add_argument("--output-dir")
    command.set_defaults(func=plot)

    command = commands.add_parser("self-check", help="run the dependency-free smoke check")
    command.set_defaults(func=self_check)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
