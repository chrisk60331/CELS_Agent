"""Semantic density maximization using rarity-weighted fact selection for optimal F1/token."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.file_summarizer import summarize_data_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_DEPTH = 7
MAX_CANDIDATES = 400


class DenseFact(BaseModel):
    """Fact with semantic density metrics."""

    label: str = Field(..., description="Field path")
    value: str = Field(..., description="Scalar value")
    semantic_density: float = Field(..., description="Information density score")
    rarity_score: float = Field(..., description="Rarity-based importance")
    uniqueness_score: float = Field(..., description="Uniqueness within document")
    tokens: int = Field(..., ge=1, description="Token count")
    depth: int = Field(..., ge=0, description="Depth in structure")


class SemanticDensityMaximizer:
    """Maximizes semantic information per token using rarity and uniqueness."""

    def __init__(self, raw_text: str, payload: Any | None):
        self.raw_text = raw_text
        self.payload = payload
        self.all_facts: list[DenseFact] = []
        self.word_frequency: Counter[str] = Counter()
        self.label_frequency: Counter[str] = Counter()
        self.value_patterns: Counter[str] = Counter()
        self._analyze_document()
        self._extract_all_facts()

    def _analyze_document(self) -> None:
        """Analyze document to build frequency models."""
        if self.payload is None:
            return
        self._count_patterns(self.payload, 0)

    def _count_patterns(self, value: Any, depth: int) -> None:
        """Count word and pattern frequencies."""
        if depth > MAX_DEPTH:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                self.label_frequency[key.lower()] += 1
                self._count_patterns(child, depth + 1)
            return

        if isinstance(value, list):
            for child in value[:20]:
                self._count_patterns(child, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if scalar_text:
            words = scalar_text.lower().split()
            self.word_frequency.update(words)
            pattern = _extract_pattern(scalar_text)
            if pattern:
                self.value_patterns[pattern] += 1

    def _extract_all_facts(self) -> None:
        """Extract all facts and compute density metrics."""
        if self.payload is None:
            return

        facts_raw: list[tuple[str, str, int]] = []
        self._walk(self.payload, (), facts_raw, 0)

        total_words = sum(self.word_frequency.values()) if self.word_frequency else 1
        total_labels = sum(self.label_frequency.values()) if self.label_frequency else 1

        for label, value, depth in facts_raw:
            rarity = self._compute_rarity(value, total_words)
            uniqueness = self._compute_uniqueness(label, value, total_labels)
            semantic_density = (rarity * 2.0 + uniqueness * 1.5) / max(1, depth + 1)
            tokens = max(1, _token_count(label) + _token_count(value))
            self.all_facts.append(
                DenseFact(
                    label=label,
                    value=value,
                    semantic_density=semantic_density,
                    rarity_score=rarity,
                    uniqueness_score=uniqueness,
                    tokens=tokens,
                    depth=depth,
                )
            )

    def _walk(self, value: Any, path: tuple[str, ...], sink: list[tuple[str, str, int]], depth: int) -> None:
        """Recursively walk JSON structure."""
        if len(sink) >= MAX_CANDIDATES or depth > MAX_DEPTH:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                self._walk(child, path + (key,), sink, depth + 1)
            return

        if isinstance(value, list):
            for idx, child in enumerate(value[:8]):
                self._walk(child, path + (f"[{idx}]",), sink, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if not scalar_text:
            return
        label = _format_path(path)
        sink.append((label, scalar_text, depth))

    def _compute_rarity(self, value: str, total_words: int) -> float:
        """Compute rarity score based on word frequency."""
        if not value or total_words == 0:
            return 0.0
        words = value.lower().split()
        if not words:
            return 0.0
        avg_frequency = sum(self.word_frequency.get(word, 0) for word in words) / len(words)
        rarity = math.log1p(total_words / max(1, avg_frequency))
        digit_bonus = 0.5 if any(ch.isdigit() for ch in value) else 0.0
        length_bonus = min(0.3, len(value) / 100.0)
        return rarity + digit_bonus + length_bonus

    def _compute_uniqueness(self, label: str, value: str, total_labels: int) -> float:
        """Compute uniqueness score based on label frequency and value patterns."""
        if total_labels == 0:
            return 0.0
        label_freq = self.label_frequency.get(label.lower(), 0)
        label_uniqueness = math.log1p(total_labels / max(1, label_freq))
        pattern = _extract_pattern(value)
        pattern_freq = self.value_patterns.get(pattern, 0) if pattern else 0
        pattern_uniqueness = math.log1p(total_labels / max(1, pattern_freq)) if pattern else 0.5
        return (label_uniqueness * 0.6 + pattern_uniqueness * 0.4)

    def select_dense_facts(self, budget: int) -> list[DenseFact]:
        """Select facts that maximize semantic density within budget."""
        sorted_facts = sorted(self.all_facts, key=lambda f: (-f.semantic_density, f.tokens, f.depth))
        selected: list[DenseFact] = []
        tokens_used = 0
        seen_combinations: set[tuple[str, str]] = set()

        for fact in sorted_facts:
            key = (fact.label.lower(), fact.value.lower()[:40])
            if key in seen_combinations:
                continue
            if tokens_used + fact.tokens > budget:
                continue
            selected.append(fact)
            tokens_used += fact.tokens
            seen_combinations.add(key)
            if len(selected) >= 25:
                break

        return selected


def run_iteration_semantic_density(task_override: str | None = None) -> AgentRunSummary:
    """Main entrypoint for semantic density maximization."""
    goal = (task_override or constants.task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty.")

    file_path = _extract_goal_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Source file not found: {full_path}")

    raw_text = full_path.read_text(encoding="utf-8")
    payload = _try_load_json(raw_text)
    base_summary = summarize_data_file(full_path, raw_text)

    maximizer = SemanticDensityMaximizer(raw_text, payload)
    word_count = _token_count(raw_text)
    budget = _compute_density_budget(word_count)
    selected_facts = maximizer.select_dense_facts(budget)

    summary_lines = [f"## SemanticDensity Summary of `{full_path.name}`"]
    avg_density = sum(f.semantic_density for f in selected_facts) / len(selected_facts) if selected_facts else 0.0
    summary_lines.append(f"High-density selection ({len(selected_facts)} facts, avg density: {avg_density:.2f})")

    for fact in selected_facts:
        summary_lines.append(f"- {fact.label}: {fact.value}")

    if len(selected_facts) < 4:
        summary_lines.append("\n### Context")
        base_lines = _extract_summary_lines(base_summary)
        summary_lines.extend(f"- {line}" for line in base_lines[:2])

    summary_text = "\n".join(summary_lines).strip()

    usage = _estimate_usage(raw_text, summary_text)
    metadata = {
        "final_answer": summary_text,
        "source_file": str(file_path),
        "fact_count": len(selected_facts),
        "token_budget": budget,
        "avg_density": avg_density,
    }

    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


def _extract_goal_file_path(goal: str) -> Path:
    """Extract file path from goal string."""
    markers = ["file ", "summarize file "]
    goal_lower = goal.lower()
    for marker in markers:
        if marker in goal_lower:
            start_idx = goal_lower.index(marker) + len(marker)
            remainder = goal[start_idx:].strip()
            if remainder:
                candidate = remainder.split()[0]
                return Path(candidate)
    raise ValueError(f"Unable to parse file path from goal: {goal}")


def _try_load_json(raw_text: str) -> Any | None:
    """Attempt to parse JSON."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def _normalize_scalar(value: Any) -> str:
    """Normalize scalar value to string."""
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        return compact[:180]
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _format_path(path: tuple[str, ...]) -> str:
    """Format path tuple to dot-separated string."""
    return ".".join(path) if path else "root"


def _token_count(text: str) -> int:
    """Estimate token count."""
    return max(1, len(text.split()))


def _extract_pattern(value: str) -> str:
    """Extract pattern signature from value."""
    if not value:
        return ""
    has_digit = any(ch.isdigit() for ch in value)
    has_alpha = any(ch.isalpha() for ch in value)
    length_cat = "short" if len(value) < 10 else "medium" if len(value) < 30 else "long"
    return f"{'d' if has_digit else ''}{'a' if has_alpha else ''}_{length_cat}"


def _compute_density_budget(word_count: int) -> int:
    """Compute token budget optimized for density."""
    if word_count < 200:
        return 70
    if word_count < 2000:
        return int(math.pow(word_count, 0.4) * 8)
    return int(math.log(word_count) * 22)


def _extract_summary_lines(summary_text: str) -> list[str]:
    """Extract meaningful lines from summary."""
    lines: list[str] = []
    for raw_line in summary_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("##"):
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if len(stripped) > 5:
            lines.append(stripped)
    return lines


def _estimate_usage(source_text: str, summary_text: str) -> dict[str, int]:
    """Estimate token usage."""
    input_tokens = _token_count(source_text)
    output_tokens = _token_count(summary_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


if __name__ == "__main__":
    summary = run_iteration_semantic_density()
    print(summary.model_dump())

