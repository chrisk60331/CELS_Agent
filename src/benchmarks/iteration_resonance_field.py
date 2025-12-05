"""ResonanceField benchmark iteration tuned for F1-per-token efficiency."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_SEGMENTS = 280
MAX_DEPTH = 10
MAX_LIST_SAMPLES = 4
MAX_OBJECT_KEYS = 12
STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "with",
    "from",
    "that",
    "this",
    "was",
    "were",
    "have",
    "has",
    "had",
    "but",
    "not",
    "into",
    "than",
    "then",
    "also",
    "such",
    "its",
    "their",
    "about",
    "between",
    "each",
    "over",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class DocumentSnapshot(BaseModel):
    """Loaded document plus metadata needed for summarization."""

    goal: str = Field(..., description="Benchmark instruction.")
    path: str = Field(..., description="Relative file path extracted from the goal.")
    raw_text: str = Field(..., description="Exact file contents.")
    payload: Any | None = Field(None, description="Parsed JSON payload when available.")
    word_count: int = Field(..., ge=0, description="Simple whitespace token count.")
    char_count: int = Field(..., ge=0, description="Character count for raw_text.")
    format_hint: str = Field(..., description="Quick descriptor for payload type.")


class Segment(BaseModel):
    """Candidate sentence for the final summary."""

    text: str = Field(..., description="Renderable content.")
    tokens: int = Field(..., ge=1, description="Estimated token cost.")
    weight: float = Field(..., ge=0.0, description="Salience weight.")
    coverage_terms: tuple[str, ...] = Field(
        ..., description="Normalized terms this segment would cover."
    )
    channel: str = Field(..., description="Origin channel for auditing.")


def run_iteration_resonance_field(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint executed by the benchmark harness."""
    goal_text = (task_override or constants.task).strip()
    if not goal_text:
        raise ValueError("Task text is empty; supply a benchmark goal.")

    snapshot = _load_document(goal_text)
    token_budget = _token_budget(snapshot.word_count)
    segments = _generate_segments(snapshot)
    summary_text = _render_summary(snapshot, segments, token_budget)
    usage = _estimate_usage(snapshot.raw_text, summary_text)
    metadata = {
        "final_answer": summary_text,
        "source_file": snapshot.path,
        "token_budget": token_budget,
        "segment_candidates": len(segments),
    }
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


def _load_document(goal_text: str) -> DocumentSnapshot:
    file_path = _extract_goal_file_path(goal_text)
    absolute_path = (PROJECT_ROOT / file_path).resolve()
    if not absolute_path.exists():
        raise FileNotFoundError(f"Source file not found: {absolute_path}")

    raw_text = absolute_path.read_text(encoding="utf-8")
    payload = _try_load_json(raw_text)
    word_count = _token_count(raw_text)
    char_count = len(raw_text)
    format_hint = _format_hint(payload)
    return DocumentSnapshot(
        goal=goal_text,
        path=str(file_path),
        raw_text=raw_text,
        payload=payload,
        word_count=word_count,
        char_count=char_count,
        format_hint=format_hint,
    )


def _generate_segments(snapshot: DocumentSnapshot) -> list[Segment]:
    structured_segments = (
        JsonSegmentEmitter(snapshot.payload).emit()
        if snapshot.payload is not None
        else []
    )
    if structured_segments:
        return structured_segments
    return TextSegmentEmitter(snapshot.raw_text).emit()


def _render_summary(snapshot: DocumentSnapshot, segments: Sequence[Segment], budget: int) -> str:
    header = f"ResonanceField digest for `{Path(snapshot.path).name}`"
    context_line = (
        f"context: format={snapshot.format_hint} words={snapshot.word_count} budget={budget}"
    )
    goal_line = f"goal: {snapshot.goal}"
    reserved_tokens = _token_count(header) + _token_count(context_line) + _token_count(goal_line)
    selection_budget = max(12, budget - reserved_tokens)
    selected = _select_segments(segments, selection_budget)
    bullet_lines = [f"- {segment.text}" for segment in selected]
    final_lines = [header, context_line, goal_line]
    final_lines.extend(bullet_lines)
    return "\n".join(line for line in final_lines if line).strip()


class JsonSegmentEmitter:
    """Walks JSON payloads to produce weighted segments."""

    def __init__(self, payload: Any):
        self._payload = payload
        self._segments: list[Segment] = []

    def emit(self) -> list[Segment]:
        self._walk(self._payload, (), 0)
        computed_segments = self._derived_segments()
        self._segments.extend(computed_segments)
        return self._segments[:MAX_SEGMENTS]

    def _walk(self, value: Any, path: tuple[str, ...], depth: int) -> None:
        if len(self._segments) >= MAX_SEGMENTS:
            return
        if depth > MAX_DEPTH:
            return

        if isinstance(value, dict):
            keys = list(value.keys())[:MAX_OBJECT_KEYS]
            descriptor = _describe_object(path, keys, len(value))
            self._record(descriptor, "object", depth)
            for key in keys:
                self._walk(value[key], path + (key,), depth + 1)
            return

        if isinstance(value, list):
            descriptor = _describe_list(path, value)
            self._record(descriptor, "list", depth)
            for idx, child in enumerate(value[:MAX_LIST_SAMPLES]):
                self._walk(child, path + (f"[{idx}]",), depth + 1)
            return

        scalar_text = _render_scalar(value)
        if not scalar_text:
            return
        label = _format_path(path)
        descriptor = f"{label}: {scalar_text}" if label else scalar_text
        self._record(descriptor, "scalar", depth)

    def _record(self, text: str, channel: str, depth: int) -> None:
        tokens = _token_count(text)
        if tokens == 0:
            return
        coverage_terms = _coverage_terms(text)
        if not coverage_terms:
            return
        weight = _segment_weight(channel, depth, coverage_terms, text)
        self._segments.append(
            Segment(
                text=text,
                tokens=tokens,
                weight=weight,
                coverage_terms=coverage_terms,
                channel=channel,
            )
        )

    def _derived_segments(self) -> list[Segment]:
        """Aggregate field frequency stats to cover global context."""
        aggregate_segments: list[Segment] = []
        label_counts = Counter()
        for segment in self._segments:
            label = segment.text.split(":", 1)[0].strip().lower()
            if not label:
                continue
            label_counts[label] += 1
        if not label_counts:
            return aggregate_segments
        most_common = label_counts.most_common(5)
        label_text = ", ".join(f"{label}×{count}" for label, count in most_common)
        descriptor = f"field density: {label_text}"
        tokens = _token_count(descriptor)
        coverage_terms = _coverage_terms(descriptor)
        if coverage_terms:
            aggregate_segments.append(
                Segment(
                    text=descriptor,
                    tokens=tokens,
                    weight=2.6,
                    coverage_terms=coverage_terms,
                    channel="computed",
                )
            )
        return aggregate_segments


class TextSegmentEmitter:
    """Extracts salient lines directly from raw text documents."""

    def __init__(self, raw_text: str):
        self._raw_text = raw_text

    def emit(self) -> list[Segment]:
        sentences = _split_text(self._raw_text)
        selection: list[Segment] = []
        seen: set[str] = set()
        for idx, sentence in enumerate(sentences):
            normalized = sentence.lower()
            if normalized in seen:
                continue
            tokens = _token_count(sentence)
            if tokens < 5:
                continue
            coverage_terms = _coverage_terms(sentence)
            if not coverage_terms:
                continue
            weight = _text_weight(idx, coverage_terms)
            selection.append(
                Segment(
                    text=sentence,
                    tokens=tokens,
                    weight=weight,
                    coverage_terms=coverage_terms,
                    channel="text",
                )
            )
            seen.add(normalized)
            if len(selection) >= MAX_SEGMENTS:
                break
        return selection


def _select_segments(segments: Sequence[Segment], budget: int) -> list[Segment]:
    available = list(segments)
    selected: list[Segment] = []
    coverage: set[str] = set()
    tokens_used = 0

    while available:
        best_idx = -1
        best_score = 0.0
        for idx, segment in enumerate(available):
            if tokens_used + segment.tokens > budget:
                continue
            new_terms = [term for term in segment.coverage_terms if term not in coverage]
            gain = len(new_terms)
            if gain == 0:
                continue
            efficiency = (gain * segment.weight) / (segment.tokens ** 0.9)
            if efficiency > best_score:
                best_score = efficiency
                best_idx = idx
        if best_idx == -1:
            break
        segment = available.pop(best_idx)
        selected.append(segment)
        tokens_used += segment.tokens
        coverage.update(segment.coverage_terms)

    if not selected and segments:
        fallback = min(segments, key=lambda item: item.tokens)
        if fallback.tokens <= budget:
            selected.append(fallback)

    return selected


def _token_budget(word_count: int) -> int:
    base = math.log1p(max(1, word_count))
    scaled = int(base * 22)
    return max(64, min(200, scaled))


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


def _format_path(path: Sequence[str]) -> str:
    return ".".join(path) if path else "root"


def _render_scalar(value: Any) -> str:
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        return compact[:220]
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _describe_object(path: Sequence[str], keys: Sequence[str], total_keys: int) -> str:
    label = _format_path(path) or "root"
    preview = ", ".join(keys[:5])
    suffix = "..." if total_keys > len(keys) else ""
    return f"{label} object keys={total_keys} ({preview}{suffix})"


def _describe_list(path: Sequence[str], values: Sequence[Any]) -> str:
    label = _format_path(path) or "root"
    size = len(values)
    samples: list[str] = []
    for item in values[:MAX_LIST_SAMPLES]:
        if isinstance(item, dict):
            sample_keys = list(item.keys())[:3]
            samples.append(f"dict[{', '.join(sample_keys)}]")
        elif isinstance(item, list):
            samples.append("list[...]")
        else:
            scalar = _render_scalar(item)
            if scalar:
                samples.append(scalar)
    sample_text = "; ".join(samples)
    return f"{label} list size={size} {sample_text}".strip()


def _split_text(raw_text: str) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer.clear()
            continue
        buffer.append(stripped)
    if buffer:
        chunks.append(" ".join(buffer))
    sentences: list[str] = []
    for chunk in chunks:
        sentences.extend(_split_sentences(chunk))
    return sentences


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _coverage_terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        if len(token) <= 2 and not token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        terms.append(token)
    ordered_terms: list[str] = []
    for token in terms:
        if token not in ordered_terms:
            ordered_terms.append(token)
    return tuple(ordered_terms[:18])


def _segment_weight(channel: str, depth: int, coverage_terms: Sequence[str], text: str) -> float:
    base = {
        "scalar": 2.4,
        "object": 1.8,
        "list": 2.0,
        "computed": 2.6,
        "text": 1.5,
    }.get(channel, 1.2)
    uniqueness = len(coverage_terms) / max(1, _token_count(text))
    numeric_bonus = 0.4 if any(ch.isdigit() for ch in text) else 0.1
    depth_factor = max(0.3, 1.3 - depth * 0.08)
    return base + uniqueness + numeric_bonus + depth_factor


def _text_weight(index: int, coverage_terms: Sequence[str]) -> float:
    novelty_bonus = len(coverage_terms) * 0.05
    position_bonus = 1.4 / (1 + index * 0.35)
    return 1.3 + novelty_bonus + position_bonus


def _token_count(text: str) -> int:
    return max(0, len(text.split()))


if __name__ == "__main__":
    summary = run_iteration_resonance_field()
    print(summary.model_dump())

