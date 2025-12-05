"""High-compression benchmark iteration focused on F1-per-token efficiency."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.file_summarizer import summarize_data_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_FACTS = 160
MAX_LIST_SAMPLES = 6


class DocumentSnapshot(BaseModel):
    """Lightweight view over the source document."""

    goal: str = Field(..., description="Benchmark instruction.")
    path: str = Field(..., description="Relative path to the source file.")
    raw_text: str = Field(..., description="Exact contents of the file.")
    payload: Any | None = Field(None, description="JSON payload when parsable.")
    word_count: int = Field(..., ge=0, description="Word count of raw_text.")
    char_count: int = Field(..., ge=0, description="Character count of raw_text.")
    format_hint: str = Field(..., description="Quick descriptor of payload type.")


class Fact(BaseModel):
    """Atomic fact extracted from the payload."""

    label: str = Field(..., description="Canonical field path.")
    text: str = Field(..., description="Scalar value rendered as text.")
    salience: float = Field(..., ge=0.0, description="Information weight.")
    tokens: int = Field(..., ge=1, description="Approximate token footprint.")


class CandidateLine(BaseModel):
    """Line that can make it into the final summary."""

    text: str = Field(..., description="Renderable text.")
    score: float = Field(..., description="Priority for selection.")
    tokens: int = Field(..., ge=1, description="Estimated token cost.")
    channel: str = Field(..., description="Origin channel for bookkeeping.")


def run_iteration_signal_blade(task_override: str | None = None) -> AgentRunSummary:
    """Main entrypoint used by the benchmark harness."""
    snapshot = _load_document(task_override or constants.task)
    base_summary = summarize_data_file(Path(snapshot.path), snapshot.raw_text)
    facts = FactExtractor(snapshot.payload).extract()
    budget = _token_budget(snapshot.word_count)
    summary_text = _compose_signal_summary(snapshot, base_summary, facts, budget)

    usage = _estimate_usage(snapshot.raw_text, summary_text)
    metadata = {
        "final_answer": summary_text,
        "source_file": snapshot.path,
        "token_budget": budget,
        "fact_count": len(facts),
    }
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


class FactExtractor:
    """Deterministic walker that pulls scalar facts out of JSON payloads."""

    def __init__(self, payload: Any | None):
        self._payload = payload

    def extract(self) -> list[Fact]:
        if self._payload is None:
            return []
        sink: list[Fact] = []
        self._walk(self._payload, (), sink, 0)
        return sink

    def _walk(self, value: Any, path: tuple[str, ...], sink: list[Fact], depth: int) -> None:
        if len(sink) >= MAX_FACTS:
            return
        if depth > 12:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                self._walk(child, path + (key,), sink, depth + 1)
            return

        if isinstance(value, list):
            for idx, child in enumerate(value[:MAX_LIST_SAMPLES]):
                self._walk(child, path + (f"[{idx}]",), sink, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if not scalar_text:
            return
        label = _format_path(path)
        tokens = max(1, _token_count(scalar_text) + max(0, len(label.split(".")) // 2))
        salience = _score_fact(label, scalar_text, depth)
        sink.append(Fact(label=label, text=scalar_text, salience=salience, tokens=tokens))


def _load_document(task_text: str) -> DocumentSnapshot:
    goal = (task_text or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; supply a goal.")
    file_path = _extract_goal_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Source file not found: {full_path}")
    raw_text = full_path.read_text(encoding="utf-8")
    payload = _try_load_json(raw_text)
    word_count = _token_count(raw_text)
    char_count = len(raw_text)
    format_hint = _format_hint(payload)
    return DocumentSnapshot(
        goal=goal,
        path=str(file_path),
        raw_text=raw_text,
        payload=payload,
        word_count=word_count,
        char_count=char_count,
        format_hint=format_hint,
    )


def _compose_signal_summary(
    snapshot: DocumentSnapshot,
    base_summary: str,
    facts: Sequence[Fact],
    budget: int,
) -> str:
    header = f"SignalBlade summary for `{Path(snapshot.path).name}`"
    meta_line = (
        f"scope: format={snapshot.format_hint}, words={snapshot.word_count}, chars={snapshot.char_count}"
    )
    candidates = list(_build_candidates(snapshot, base_summary, facts))
    selection_budget = budget - _token_count(header) - _token_count(meta_line)
    selected = _select_candidates(candidates, max(1, selection_budget))
    bullet_lines = [f"- {line.text}" if not line.text.startswith("-") else line.text for line in selected]
    final_lines = [header, meta_line]
    final_lines.extend(bullet_lines)
    return "\n".join(final_lines).strip()


def _build_candidates(
    snapshot: DocumentSnapshot,
    base_summary: str,
    facts: Sequence[Fact],
) -> Iterable[CandidateLine]:
    yield CandidateLine(
        text=f"goal: {snapshot.goal[:180]}",
        score=3.0,
        tokens=max(1, _token_count(snapshot.goal[:180])),
        channel="goal",
    )
    for fact in facts:
        text = f"{fact.label}: {fact.text}"
        yield CandidateLine(text=text, score=2.5 + fact.salience, tokens=fact.tokens, channel="fact")
    for idx, line in enumerate(_extract_summary_lines(base_summary)):
        tokens = _token_count(line)
        if tokens == 0:
            continue
        score = max(0.1, 1.2 - idx * 0.03)
        yield CandidateLine(text=line, score=score, tokens=tokens, channel="summary")


def _select_candidates(candidates: Iterable[CandidateLine], budget: int) -> list[CandidateLine]:
    ordered = sorted(candidates, key=lambda item: (-item.score, item.tokens))
    selected: list[CandidateLine] = []
    tokens_used = 0
    seen: set[str] = set()
    for candidate in ordered:
        normalized = candidate.text.lower()
        if normalized in seen:
            continue
        if tokens_used + candidate.tokens > budget:
            continue
        selected.append(candidate)
        seen.add(normalized)
        tokens_used += candidate.tokens
    return selected


def _extract_summary_lines(summary_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in summary_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("##"):
            continue
        lowered = stripped.lower()
        if lowered in {"key facts:", "numeric highlights:", "keywords:"}:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        lines.append(stripped)
    return lines


def _token_budget(word_count: int) -> int:
    base = math.sqrt(max(1, word_count))
    budget = int(base * 5)
    return max(80, min(220, budget))


def _estimate_usage(source_text: str, summary_text: str) -> dict[str, int]:
    input_tokens = _token_count(source_text)
    output_tokens = _token_count(summary_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _extract_goal_file_path(goal: str) -> Path:
    marker = "file "
    goal_lower = goal.lower()
    if marker not in goal_lower:
        raise ValueError(f"Goal must include 'file <path>': {goal}")
    start_idx = goal_lower.index(marker) + len(marker)
    remainder = goal[start_idx:].strip()
    if not remainder:
        raise ValueError(f"Unable to parse file path from goal: {goal}")
    candidate = remainder.split()[0]
    return Path(candidate)


def _try_load_json(raw_text: str) -> Any | None:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def _format_hint(payload: Any | None) -> str:
    if isinstance(payload, dict):
        return f"json.object keys={len(payload)}"
    if isinstance(payload, list):
        return f"json.list items={len(payload)}"
    return "text"


def _normalize_scalar(value: Any) -> str:
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        return compact[:220]
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _format_path(path: Sequence[str]) -> str:
    return ".".join(path) if path else "root"


def _score_fact(label: str, text: str, depth: int) -> float:
    value_tokens = text.split()
    uniqueness = len(set(token.lower() for token in value_tokens)) / max(1, len(value_tokens))
    numeric_bonus = 1.0 if any(ch.isdigit() for ch in text) else 0.3
    label_bonus = 0.4 if any(ch.isdigit() for ch in label) else 0.0
    depth_penalty = 1.0 + depth * 0.15
    return (1.2 + uniqueness + numeric_bonus + label_bonus) / depth_penalty


def _token_count(text: str) -> int:
    return max(0, len(text.split()))


if __name__ == "__main__":
    summary = run_iteration_signal_blade()
    print(summary.model_dump())


