from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from time import perf_counter
from typing import Callable, Iterable, Sequence
import logging
from pydantic import BaseModel, Field

from benchmarks.agent_run_summary import AgentRunSummary
from benchmarks.main import run_default_agent
from benchmarks.main_toon import run_toon_agent
from benchmarks.main_compressed_agent import run_compressed_agent
from benchmarks.main_llmlingua import run_llmlingua_agent
from benchmarks.main_agentprune import run_agentprune_agent

RunnerFn = Callable[[str | None], AgentRunSummary]
logging.basicConfig(level=logging.WARN)


class BenchmarkResult(BaseModel):
    """Timing and usage summary for a single agent entrypoint."""

    name: str
    duration_seconds: float
    summary: AgentRunSummary = Field(..., description="Usage metrics emitted by the run.")


class BenchmarkScoreResult(BenchmarkResult):
    """Benchmark result augmented with an F1 score for response quality."""

    f1_score: float = Field(..., ge=0.0, le=1.0)


def _run_benchmark(name: str, runner: RunnerFn, task_override: str | None) -> BenchmarkResult:
    start = perf_counter()
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        summary = runner(task_override)
    if summary is None:
        raise RuntimeError(f"Runner '{name}' did not produce a summary.")
    if name in ["compressed_agent", "main"]:
        logging.info(f"{name} summary: {summary}")
    

    elapsed = perf_counter() - start
    return BenchmarkResult(name=name, duration_seconds=elapsed, summary=summary)


def run_benchmarks(task_override: str | None = None) -> list[BenchmarkResult]:
    """Execute all entrypoints and collect timing plus token usage."""
    runners: Sequence[tuple[str, RunnerFn]] = [
        ("main", run_default_agent),
        ("main_toon", run_toon_agent),
        ("compressed_agent", run_compressed_agent),
        ("llmlingua_agent", run_llmlingua_agent),
        ("agentprune_agent", run_agentprune_agent),
    ]
    return [_run_benchmark(name, runner, task_override) for name, runner in runners]


def display_results(results: Iterable[BenchmarkResult]) -> None:
    """Print a concise view of benchmark outcomes."""
    for result in results:
        total_tokens = result.summary.total_tokens
        usage = result.summary.usage or {}
        print(
            f"{result.name}: {result.duration_seconds:.3f}s | "
            f"total_tokens={total_tokens} | usage={usage}\n\n"
        )


def benchmark_with_score(reference_output: str, task_instruction: str | None = None) -> list[BenchmarkScoreResult]:
    """Run benchmarks and score their outputs against a reference answer."""
    if not isinstance(reference_output, str) or not reference_output.strip():
        raise ValueError("reference_output must be a non-empty string.")

    benchmarks = run_benchmarks(task_instruction)
    return [
        BenchmarkScoreResult(
            name=result.name,
            duration_seconds=result.duration_seconds,
            summary=result.summary,
            f1_score=_f1_score(reference_output, _extract_final_answer(result)),
        )
        for result in benchmarks
    ]


def _extract_final_answer(result: BenchmarkResult) -> str:
    final_answer = result.summary.metadata.get("final_answer")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError(
            f"Benchmark '{result.name}' did not record a final_answer in summary metadata."
        )
    return final_answer.strip()


def _f1_score(reference: str, candidate: str) -> float:
    reference_tokens = _tokenize(reference)
    candidate_tokens = _tokenize(candidate)

    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)

    true_positive = sum(
        min(candidate_counts[token], reference_counts[token]) for token in candidate_counts
    )
    predicted_total = sum(candidate_counts.values())
    reference_total = sum(reference_counts.values())

    if predicted_total == 0 or reference_total == 0:
        return 0.0

    precision = true_positive / predicted_total
    recall = true_positive / reference_total
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _tokenize(value: str) -> list[str]:
    tokens = [part for part in value.lower().split() if part]
    if not tokens:
        return []
    return tokens


if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    display_results(benchmark_results)

