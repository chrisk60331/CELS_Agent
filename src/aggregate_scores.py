from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field
from tabulate import tabulate

from benchmark import BenchmarkScoreResult, benchmark_with_score
import constants


class AgentAggregate(BaseModel):
    """Aggregate F1 scores and token usage across the configured test suite."""

    name: str = Field(..., description="Agent entrypoint identifier.")
    runs: int = Field(..., ge=1, description="Number of benchmarked tests.")
    average_f1: float = Field(..., ge=0.0, le=1.0)
    average_duration_seconds: float = Field(..., ge=0.0)
    total_duration_seconds: float = Field(..., ge=0.0)
    runs_with_token_usage: int = Field(..., ge=0)
    average_total_tokens: float | None = Field(default=None, ge=0.0)
    total_tokens: int | None = Field(default=None, ge=0)

    @classmethod
    def from_results(cls, name: str, results: Sequence[BenchmarkScoreResult]) -> "AgentAggregate":
        if not results:
            raise ValueError(f"No benchmark results provided for agent '{name}'.")

        runs = len(results)
        total_duration = sum(item.duration_seconds for item in results)
        total_f1 = sum(item.f1_score for item in results)
        token_values = [
            item.summary.total_tokens for item in results if item.summary.total_tokens is not None
        ]

        total_tokens = sum(token_values) if token_values else None
        average_tokens = (total_tokens / len(token_values)) if token_values else None

        return cls(
            name=name,
            runs=runs,
            average_f1=total_f1 / runs,
            average_duration_seconds=total_duration / runs,
            total_duration_seconds=total_duration,
            runs_with_token_usage=len(token_values),
            average_total_tokens=average_tokens,
            total_tokens=total_tokens,
        )


def _load_reference(reference_path: Path) -> str:
    """Read and validate the reference output for a test case."""
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference summary not found at '{reference_path}'.")

    reference_text = reference_path.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise ValueError(f"Reference summary at '{reference_path}' is empty.")
    return reference_text


def _accumulate(results: Iterable[BenchmarkScoreResult], store: dict[str, list[BenchmarkScoreResult]]) -> None:
    for result in results:
        store[result.name].append(result)


def main() -> None:
    if not constants.tests:
        raise ValueError("No tests configured in 'constants.tests'.")

    base_dir = Path(__file__).resolve().parent
    aggregates: dict[str, list[BenchmarkScoreResult]] = defaultdict(list)
    for task_instruction, reference_rel_path in constants.tests:
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ValueError("Each test must define a non-empty task instruction.")
        if not isinstance(reference_rel_path, str) or not reference_rel_path.strip():
            raise ValueError("Each test must provide a non-empty reference path.")

        goal = task_instruction.strip()
        # Ensure all agents have the same number of runs by only benchmarking
        # file-based summarization tasks (those that include a source file path).
        if "summarize file " not in goal.lower():
            print(f"\nSkipping non-file task: {goal}")
            continue
        reference_path = (base_dir / reference_rel_path).resolve()
        reference_text = _load_reference(reference_path)

        results = benchmark_with_score(reference_text, goal)
        results.sort(key=lambda item: (-item.f1_score, item.duration_seconds))

        print(f"\nTask: {goal}")
        for result in results:
            total_tokens = result.summary.total_tokens
            print(
                f"  {result.name}: f1={result.f1_score:.3f} | "
                f"duration={result.duration_seconds:.3f}s | "
                f"total_tokens={total_tokens if total_tokens is not None else 'n/a'}"
            )

        _accumulate(results, aggregates)

    if not aggregates:
        raise RuntimeError("Benchmark execution produced no results.")

    summary = [
        AgentAggregate.from_results(name, agent_results)
        for name, agent_results in aggregates.items()
    ]
    summary.sort(key=lambda item: (-item.average_f1, item.average_duration_seconds))

    print("\n=== Aggregated Scores ===")
    table_rows = []
    for aggregate in summary:
        table_rows.append(
            [
                aggregate.name,
                aggregate.runs,
                f"{aggregate.average_f1:.3f}",
                f"{aggregate.average_duration_seconds:.3f}",
                f"{aggregate.total_duration_seconds:.3f}",
                (
                    f"{aggregate.average_total_tokens:.1f}"
                    if aggregate.average_total_tokens is not None
                    else "n/a"
                ),
                aggregate.total_tokens if aggregate.total_tokens is not None else "n/a",
            ]
        )

    headers = [
        "Agent",
        "Runs",
        "Avg F1",
        "Avg Duration (s)",
        "Total Duration (s)",
        "Avg Tokens",
        "Total Tokens",
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()


