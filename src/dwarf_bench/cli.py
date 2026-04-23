"""Command-line entry point for dwarf-bench."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from dwarf_bench.dataset import load_questions
from dwarf_bench.judge import DEFAULT_JUDGE_MODEL, grade_artifact
from dwarf_bench.providers import AnthropicProvider
from dwarf_bench.results import RunArtifact
from dwarf_bench.runner import run_benchmark

DEFAULT_DATASET = "data/questions.jsonl"
DEFAULT_RESULTS_DIR = "results"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="dwarf-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Query a model on the dataset")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--dataset", default=DEFAULT_DATASET)
    p_run.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p_run.add_argument("--concurrency", type=int, default=5)

    p_grade = sub.add_parser("grade", help="Grade an existing run file")
    p_grade.add_argument("run_file")
    p_grade.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p_grade.add_argument("--concurrency", type=int, default=5)

    p_bench = sub.add_parser("bench", help="Run + grade one or more models")
    p_bench.add_argument("--models", required=True, help="Comma-separated list")
    p_bench.add_argument("--dataset", default=DEFAULT_DATASET)
    p_bench.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p_bench.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p_bench.add_argument("--concurrency", type=int, default=5)

    p_report = sub.add_parser("report", help="Summarize graded runs")
    p_report.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return asyncio.run(_cmd_run(args))
    if args.cmd == "grade":
        return asyncio.run(_cmd_grade(args))
    if args.cmd == "bench":
        return asyncio.run(_cmd_bench(args))
    if args.cmd == "report":
        return _cmd_report(args)
    return 2


async def _cmd_run(args) -> int:
    questions = load_questions(args.dataset)
    provider = AnthropicProvider()
    print(f"Running {len(questions)} questions on {args.model}...")
    artifact = await run_benchmark(
        provider=provider,
        model=args.model,
        questions=questions,
        concurrency=args.concurrency,
    )
    path = artifact.save(args.results_dir)
    errors = sum(1 for r in artifact.results if r.error)
    print(f"Saved {path} ({len(artifact.results)} results, {errors} errors)")
    return 0


async def _cmd_grade(args) -> int:
    artifact = RunArtifact.load(args.run_file)
    judge = AnthropicProvider()
    print(f"Grading {args.run_file} with {args.judge_model}...")
    await grade_artifact(
        artifact,
        judge=judge,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
    )
    artifact.save(Path(args.run_file).parent)
    acc = artifact.accuracy()
    print(f"Accuracy: {acc:.2%}" if acc is not None else "No grades produced.")
    return 0


async def _cmd_bench(args) -> int:
    questions = load_questions(args.dataset)
    provider = AnthropicProvider()
    judge = AnthropicProvider()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows: list[tuple[str, float | None, int]] = []
    for model in models:
        print(f"\n=== {model} ===")
        print(f"  Running {len(questions)} questions...")
        artifact = await run_benchmark(
            provider=provider,
            model=model,
            questions=questions,
            concurrency=args.concurrency,
        )
        print(f"  Grading with {args.judge_model}...")
        await grade_artifact(
            artifact,
            judge=judge,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
        )
        path = artifact.save(args.results_dir)
        errors = sum(1 for r in artifact.results if r.error)
        acc = artifact.accuracy()
        print(f"  Saved {path}")
        print(f"  Accuracy: {acc:.2%}" if acc is not None else "  No grades.")
        rows.append((model, acc, errors))
    _print_table(rows)
    return 0


def _cmd_report(args) -> int:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"No results directory at {results_dir}", file=sys.stderr)
        return 1
    rows: list[tuple[str, float | None, int]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        try:
            artifact = RunArtifact.load(path)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
            continue
        errors = sum(1 for r in artifact.results if r.error)
        rows.append((f"{artifact.model} ({artifact.run_id})", artifact.accuracy(), errors))
    _print_table(rows)
    return 0


def _print_table(rows: list[tuple[str, float | None, int]]) -> None:
    if not rows:
        print("(no runs)")
        return
    width = max(len(name) for name, _, _ in rows)
    print(f"\n{'model'.ljust(width)}   accuracy   errors")
    print(f"{'-' * width}   --------   ------")
    for name, acc, errors in rows:
        acc_s = f"{acc:.2%}" if acc is not None else "    —  "
        print(f"{name.ljust(width)}   {acc_s:>8}   {errors:>6}")


if __name__ == "__main__":
    raise SystemExit(main())
