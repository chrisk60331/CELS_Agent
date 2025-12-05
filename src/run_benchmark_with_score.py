from __future__ import annotations

from pathlib import Path

from tabulate import tabulate

from benchmark import BenchmarkScoreResult, benchmark_with_score
from constants import SUMMARY_FILE

SUMMARY_PATH = Path(__file__).resolve().parent / SUMMARY_FILE


def _f1_per_token(result: BenchmarkScoreResult) -> float:
    total_tokens = result.summary.total_tokens
    if total_tokens is None or total_tokens <= 0:
        return 0.0
    return result.f1_score / total_tokens


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Reference summary file not found at {SUMMARY_PATH}. Run documentation to regenerate."
        )

    reference_text = SUMMARY_PATH.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise ValueError(f"Reference summary at {SUMMARY_PATH} is empty.")

    results = benchmark_with_score(reference_text)
    # Sort by F1 per token descending to highlight efficient runs first.
    results.sort(key=_f1_per_token, reverse=True)

    table_rows: list[list[str]] = []
    for result in results:
        total_tokens = result.summary.total_tokens
        f1_per_token = (
            result.f1_score / total_tokens
            if total_tokens is not None and total_tokens > 0
            else None
        )
        f1_per_token_display = (
            f"{f1_per_token:.6f}" if f1_per_token is not None else "n/a"
        )
        table_rows.append(
            [
                result.name,
                f1_per_token_display,
                f"{result.f1_score:.3f}",
                f"{result.duration_seconds:.3f}",
                str(total_tokens) if total_tokens is not None else "n/a",
            ]
        )

    headers = ["name", "f1/token", "f1", "duration (s)", "total tokens"]

    print(f"Reference summary loaded from: {SUMMARY_PATH}")
    print(
        tabulate(
            table_rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
        )
    )


if __name__ == "__main__":
    main()

