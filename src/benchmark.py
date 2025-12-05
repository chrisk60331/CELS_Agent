from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from time import perf_counter
from typing import Callable, Iterable, NamedTuple, Sequence
import logging
from pydantic import BaseModel, Field

from agent_run_summary import AgentRunSummary
from benchmarks.main import run_default_agent

from benchmarks.recursive_summarizer import run_recursive_summarizer
from benchmarks.iteration_nucleus_extractor import run_iteration_nucleus_extractor
from benchmarks.iteration_fractal_adaptive import run_iteration_fractal_adaptive
from benchmarks.iteration_selective_cascade import run_iteration_selective_cascade
from benchmarks.iteration_precision_lens import run_iteration_precision_lens
from benchmarks.iteration_budget_rag import run_iteration_budget_rag
from benchmarks.iteration_temporal_synth import run_iteration_temporal_synth
from benchmarks.main_compressed_agent import run_compressed_agent
import constants

RunnerFn = Callable[[str | None], AgentRunSummary]
logging.basicConfig(level=logging.WARN)


class RunnerSpec(NamedTuple):
    name: str
    runner: RunnerFn
    requires_file_goal: bool = False


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
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            summary = runner(task_override)
        summary = json.loads(summary.model_dump_json())
    except Exception as exc:
        logging.error(exc)
        summary = AgentRunSummary()
    logging.warning(f"{name} summary type: {type(summary)}")
    logging.warning(f"{name} summary type: {summary}")
    if name in ["recursive_summarizer", "main"]:
        pass

    elapsed = perf_counter() - start
    return BenchmarkResult(name=name, duration_seconds=elapsed, summary=summary)


RUNNER_SPECS: Sequence[RunnerSpec] = [
    RunnerSpec("run_compressed_agent", run_compressed_agent, True),
    RunnerSpec("main", run_default_agent, False),
    RunnerSpec("recursive_summarizer", run_recursive_summarizer, True),
    RunnerSpec("iteration_temporal_synth", run_iteration_temporal_synth, True),
    # RunnerSpec("iteration_nucleus_extractor", run_iteration_nucleus_extractor, True),
    # RunnerSpec("iteration_fractal_adaptive", run_iteration_fractal_adaptive, True),
    # RunnerSpec("iteration_selective_cascade", run_iteration_selective_cascade, True),
    # RunnerSpec("iteration_precision_lens", run_iteration_precision_lens, True),
    # RunnerSpec("iteration_budget_rag", run_iteration_budget_rag, True),
]


def run_benchmarks(task_override: str | None = None) -> list[BenchmarkResult]:
    """Execute all entrypoints and collect timing plus token usage."""
    goal_requires_file = _goal_requires_file(task_override)
    results: list[BenchmarkResult] = []
    for spec in RUNNER_SPECS:
        if spec.requires_file_goal and not goal_requires_file:
            logging.info(
                "Skipping benchmark '%s' because the task does not reference a source file.",
                spec.name,
            )
            continue
        results.append(_run_benchmark(spec.name, spec.runner, task_override))

    if not results:
        raise ValueError("No benchmarks were eligible to run for the provided task.")
    return results


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


def _goal_requires_file(task_override: str | None) -> bool:
    goal = _normalize_goal(task_override)
    return "summarize file " in goal.lower()


def _normalize_goal(task_override: str | None) -> str:
    if isinstance(task_override, str) and task_override.strip():
        return task_override.strip()
    return (getattr(constants, "task", "") or "").strip()


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

