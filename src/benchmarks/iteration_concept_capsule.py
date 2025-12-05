"""Concept Capsule iteration tuned for extreme F1-per-token efficiency."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_CAPSULES = 220
MAX_DEPTH = 8
MAX_LIST_SAMPLES = 4
MAX_CHILDREN = 60
MAX_TEXT_SENTENCES = 240
MIN_SENTENCE_TOKENS = 6
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
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
    "per",
    "after",
    "before",
    "any",
}


class DocumentSnapshot(BaseModel):
    """Canonical view over the benchmark document."""

    goal: str = Field(..., description="Benchmark instruction.")
    path: str = Field(..., description="Relative source path extracted from the goal.")
    raw_text: str = Field(..., description="Exact file contents.")
    payload: Any | None = Field(None, description="Parsed JSON payload when available.")
    word_count: int = Field(..., ge=0, description="Whitespace token count of the file.")
    char_count: int = Field(..., ge=0, description="Character length of the file.")
    format_hint: str = Field(..., description="Quick descriptor for downstream formatting.")


class ConceptCapsule(BaseModel):
    """Atomic fragment competing for space in the final summary."""

    text: str = Field(..., description="Renderable content.")
    tokens: int = Field(..., ge=1, description="Estimated token footprint.")
    weight: float = Field(..., ge=0.0, description="Information weight.")
    coverage_terms: tuple[str, ...] = Field(
        ..., description="Normalized terms this capsule would cover."
    )
    scope: str = Field(..., description="Top-level scope label.")
    origin: str = Field(..., description="Channel identifier for auditing.")


def run_iteration_concept_capsule(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint executed by the benchmark harness."""
    goal_text = (task_override or constants.task).strip()
    if not goal_text:
        raise ValueError("Benchmark task is empty; supply a goal.")

    snapshot = _load_document(goal_text)
    capsule_pool = CapsulePlanner(snapshot).build()
    if not capsule_pool:
        capsule_pool = [_fallback_capsule(snapshot)]
    budget = _token_budget(snapshot.word_count)
    summary_text = _compose_summary(snapshot, capsule_pool, budget)
    usage = _estimate_usage(snapshot.raw_text, summary_text)
    metadata = {
        "final_answer": summary_text,
        "source_file": snapshot.path,
        "token_budget": budget,
        "capsule_candidates": len(capsule_pool),
    }
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


class CapsulePlanner:
    """Chooses the correct extractor based on payload type."""

    def __init__(self, snapshot: DocumentSnapshot):
        self._snapshot = snapshot

    def build(self) -> list[ConceptCapsule]:
        if self._snapshot.payload is not None:
            builder = JsonCapsuleBuilder(self._snapshot.payload)
            return builder.build()
        builder = PlaintextCapsuleBuilder(self._snapshot.raw_text)
        return builder.build()


class JsonCapsuleBuilder:
    """Extracts high-density capsules from JSON payloads."""

    def __init__(self, payload: Any):
        self._payload = payload
        self._capsules: list[ConceptCapsule] = []
        self._scope_counts: Counter[str] = Counter()

    def build(self) -> list[ConceptCapsule]:
        self._walk(self._payload, (), 0)
        self._capsules.extend(self._scope_density_capsules())
        return self._capsules[:MAX_CAPSULES]

    def _walk(self, value: Any, path: tuple[str, ...], depth: int) -> None:
        if len(self._capsules) >= MAX_CAPSULES:
            return
        if depth > MAX_DEPTH:
            return

        if isinstance(value, dict):
            self._record_object(value, path, depth)
            for key in list(value.keys())[:MAX_CHILDREN]:
                self._walk(value[key], path + (key,), depth + 1)
            return

        if isinstance(value, list):
            self._record_list(value, path, depth)
            for idx, child in enumerate(value[:MAX_LIST_SAMPLES]):
                self._walk(child, path + (f"[{idx}]",), depth + 1)
            return

        self._record_scalar(value, path, depth)

    def _record_object(self, obj: dict[str, Any], path: tuple[str, ...], depth: int) -> None:
        descriptor = _describe_object(path, obj)
        self._record_capsule(descriptor, path, depth, "object")

    def _record_list(self, values: Sequence[Any], path: tuple[str, ...], depth: int) -> None:
        descriptor = _describe_list(path, values)
        self._record_capsule(descriptor, path, depth, "list")

    def _record_scalar(self, value: Any, path: tuple[str, ...], depth: int) -> None:
        descriptor = _describe_scalar(path, value)
        if descriptor:
            self._record_capsule(descriptor, path, depth, "scalar")

    def _record_capsule(self, text: str, path: tuple[str, ...], depth: int, channel: str) -> None:
        tokens = _token_count(text)
        if tokens == 0:
            return
        coverage_terms = _coverage_terms(text)
        if not coverage_terms:
            return
        scope = _scope_from_path(path)
        weight = _capsule_weight(channel, depth, text, len(coverage_terms))
        capsule = ConceptCapsule(
            text=text,
            tokens=tokens,
            weight=weight,
            coverage_terms=coverage_terms,
            scope=scope,
            origin=channel,
        )
        self._capsules.append(capsule)
        self._scope_counts[scope] += 1

    def _scope_density_capsules(self) -> list[ConceptCapsule]:
        if not self._scope_counts:
            return []
        aggregate_capsules: list[ConceptCapsule] = []
        most_common = self._scope_counts.most_common(6)
        summary_text = ", ".join(f"{scope}×{count}" for scope, count in most_common)
        descriptor = f"field density: {summary_text}"
        coverage_terms = _coverage_terms(descriptor)
        if not coverage_terms:
            return []
        capsule = ConceptCapsule(
            text=descriptor,
            tokens=_token_count(descriptor),
            weight=3.2,
            coverage_terms=coverage_terms,
            scope="structure",
            origin="aggregate",
        )
        aggregate_capsules.append(capsule)
        return aggregate_capsules


class PlaintextCapsuleBuilder:
    """Extracts high-salience sentences from raw text documents."""

    def __init__(self, raw_text: str):
        self._raw_text = raw_text

    def build(self) -> list[ConceptCapsule]:
        sentences = _split_sentences(self._raw_text)
        if not sentences:
            return []
        sentence_data = []
        for idx, sentence in enumerate(sentences[:MAX_TEXT_SENTENCES]):
            tokens = _token_count(sentence)
            if tokens < MIN_SENTENCE_TOKENS:
                continue
            coverage_terms = _coverage_terms(sentence)
            if not coverage_terms:
                continue
            sentence_data.append((idx, sentence, tokens, coverage_terms))
        if not sentence_data:
            return []
        term_counts = Counter(term for _, _, _, terms in sentence_data for term in terms)
        capsules: list[ConceptCapsule] = []
        for idx, sentence, tokens, coverage_terms in sentence_data:
            novelty = sum(1.0 / (1 + term_counts[term]) for term in coverage_terms)
            positional_bonus = 0.6 if idx < 6 else 0.2 if idx < 18 else 0.0
            weight = 2.1 + novelty + positional_bonus
            capsules.append(
                ConceptCapsule(
                    text=sentence,
                    tokens=tokens,
                    weight=weight,
                    coverage_terms=coverage_terms,
                    scope=f"text_{idx // 5}",
                    origin="text",
                )
            )
            if len(capsules) >= MAX_CAPSULES:
                break
        return capsules


def _compose_summary(
    snapshot: DocumentSnapshot,
    capsules: Sequence[ConceptCapsule],
    budget: int,
) -> str:
    selector = CapsuleSelector(budget)
    selected = selector.select(capsules)
    if not selected:
        fallback_line = _fallback_capsule(snapshot).text
        selected = [
            ConceptCapsule(
                text=fallback_line,
                tokens=_token_count(fallback_line),
                weight=1.0,
                coverage_terms=_coverage_terms(fallback_line),
                scope="fallback",
                origin="fallback",
            )
        ]

    header = f"ConceptCapsule digest for `{Path(snapshot.path).name}`"
    context = (
        f"format={snapshot.format_hint} words={snapshot.word_count} "
        f"chars={snapshot.char_count} budget={budget}"
    )
    goal_line = f"goal: {snapshot.goal}"
    bullets = [f"- {capsule.text}" for capsule in selected]
    lines = [header, context, goal_line]
    lines.extend(bullets)
    return "\n".join(line for line in lines if line).strip()


class CapsuleSelector:
    """Greedy selector that maximizes coverage gain per token."""

    def __init__(self, budget: int):
        if budget <= 0:
            raise ValueError("Token budget must be positive.")
        self._budget = budget

    def select(self, capsules: Sequence[ConceptCapsule]) -> list[ConceptCapsule]:
        available = list(capsules)
        selected: list[ConceptCapsule] = []
        tokens_used = 0
        coverage: set[str] = set()

        for scope, scoped_capsules in _group_by_scope(available).items():
            best = max(
                scoped_capsules,
                key=lambda capsule: capsule.weight / max(1, capsule.tokens),
                default=None,
            )
            if best is None:
                continue
            if tokens_used + best.tokens > self._budget:
                continue
            selected.append(best)
            tokens_used += best.tokens
            coverage.update(best.coverage_terms)
            available.remove(best)
            if tokens_used >= self._budget:
                return selected

        while available:
            best_idx = -1
            best_score = 0.0
            for idx, capsule in enumerate(available):
                if tokens_used + capsule.tokens > self._budget:
                    continue
                new_terms = [term for term in capsule.coverage_terms if term not in coverage]
                if not new_terms:
                    continue
                gain = len(new_terms)
                score = (capsule.weight * gain) / (capsule.tokens ** 0.85)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx == -1:
                break
            capsule = available.pop(best_idx)
            selected.append(capsule)
            tokens_used += capsule.tokens
            coverage.update(capsule.coverage_terms)
            if tokens_used >= self._budget:
                break

        if not selected and capsules:
            fallback = min(capsules, key=lambda capsule: capsule.tokens)
            if fallback.tokens <= self._budget:
                selected.append(fallback)
        return selected


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


def _token_budget(word_count: int) -> int:
    scaled = int((math.sqrt(max(1, word_count)) * 1.8) + 32)
    return max(64, min(180, scaled))


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
    lowered = goal.lower()
    if marker not in lowered:
        raise ValueError(f"Goal must include 'file <path>': {goal}")
    start_idx = lowered.index(marker) + len(marker)
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


def _describe_object(path: Sequence[str], value: dict[str, Any]) -> str:
    label = _format_path(path) or "root"
    keys = list(value.keys())
    preview = ", ".join(keys[:5])
    suffix = "..." if len(keys) > 5 else ""
    return f"{label} object keys={len(value)} ({preview}{suffix})"


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
            scalar = _format_scalar(item)
            if scalar:
                samples.append(scalar)
    suffix = f" samples: {'; '.join(samples)}" if samples else ""
    return f"{label} list size={size}{suffix}"


def _describe_scalar(path: Sequence[str], value: Any) -> str:
    label = _format_path(path) or "value"
    scalar = _format_scalar(value)
    if not scalar:
        return ""
    ending = "" if scalar.endswith((".", "!", "?")) else "."
    return f"{label}: {scalar}{ending}"


def _format_path(path: Sequence[str]) -> str:
    return ".".join(path) if path else ""


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        if not compact:
            return ""
        limit = 160
        return compact[:limit] + ("..." if len(compact) > limit else "")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _scope_from_path(path: Sequence[str]) -> str:
    for segment in path:
        if segment and not segment.startswith("["):
            return segment
    return "root"


def _capsule_weight(channel: str, depth: int, text: str, term_count: int) -> float:
    base = {
        "object": 2.0,
        "list": 2.2,
        "scalar": 2.8,
        "aggregate": 3.2,
        "text": 2.4,
        "fallback": 1.0,
    }.get(channel, 2.0)
    numeric_bonus = 0.6 if any(ch.isdigit() for ch in text) else 0.0
    depth_penalty = 1.0 + depth * 0.12
    density_bonus = min(1.2, term_count * 0.08)
    return (base + numeric_bonus + density_bonus) / depth_penalty


def _coverage_terms(text: str) -> tuple[str, ...]:
    tokens = [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token and token.lower() not in STOPWORDS
    ]
    unique_terms: dict[str, None] = {}
    for token in tokens:
        if len(token) < 3:
            continue
        unique_terms.setdefault(token, None)
    return tuple(unique_terms.keys())


def _split_sentences(raw_text: str) -> list[str]:
    sanitized = " ".join(raw_text.replace("\r", " ").split())
    if not sanitized:
        return []
    return [segment.strip() for segment in SENTENCE_SPLIT.split(sanitized) if segment.strip()]


def _group_by_scope(capsules: Iterable[ConceptCapsule]) -> dict[str, list[ConceptCapsule]]:
    grouped: dict[str, list[ConceptCapsule]] = defaultdict(list)
    for capsule in capsules:
        grouped[capsule.scope].append(capsule)
    return grouped


def _fallback_capsule(snapshot: DocumentSnapshot) -> ConceptCapsule:
    excerpt = snapshot.raw_text[:220].replace("\n", " ").strip()
    if not excerpt:
        excerpt = "source document is empty."
    text = f"raw excerpt: {excerpt}"
    coverage_terms = _coverage_terms(text)
    if not coverage_terms:
        coverage_terms = ("excerpt",)
    return ConceptCapsule(
        text=text,
        tokens=_token_count(text),
        weight=1.5,
        coverage_terms=coverage_terms,
        scope="fallback",
        origin="fallback",
    )


def _token_count(text: str) -> int:
    return max(0, len(text.split()))


if __name__ == "__main__":
    summary = run_iteration_concept_capsule()
    print(summary.model_dump())


