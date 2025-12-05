"""Adaptive token budget allocation based on document structure and information density."""
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


class AdaptiveFact(BaseModel):
    """Fact with adaptive scoring metrics."""

    label: str = Field(..., description="Field path")
    value: str = Field(..., description="Scalar value")
    information_value: float = Field(..., description="Information value score")
    structural_importance: float = Field(..., description="Structural importance")
    adaptive_score: float = Field(..., description="Combined adaptive score")
    tokens: int = Field(..., ge=1, description="Token count")
    depth: int = Field(..., ge=0, description="Depth in structure")


class AdaptiveBudgetAllocator:
    """Adaptively allocates token budget based on document characteristics."""

    def __init__(self, raw_text: str, payload: Any | None):
        self.raw_text = raw_text
        self.payload = payload
        self.all_facts: list[AdaptiveFact] = []
        self.structure_stats: dict[str, int] = {
            "total_keys": 0,
            "total_lists": 0,
            "max_depth": 0,
            "scalar_count": 0,
        }
        self.value_diversity: Counter[str] = Counter()
        self._analyze_structure()
        self._extract_all_facts()

    def _analyze_structure(self) -> None:
        """Analyze document structure to inform budget allocation."""
        if self.payload is None:
            return
        self._walk_structure(self.payload, 0)

    def _walk_structure(self, value: Any, depth: int) -> None:
        """Walk structure to collect statistics."""
        self.structure_stats["max_depth"] = max(self.structure_stats["max_depth"], depth)
        if depth > MAX_DEPTH:
            return

        if isinstance(value, dict):
            self.structure_stats["total_keys"] += len(value)
            for child in value.values():
                self._walk_structure(child, depth + 1)
            return

        if isinstance(value, list):
            self.structure_stats["total_lists"] += 1
            for child in value[:10]:
                self._walk_structure(child, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if scalar_text:
            self.structure_stats["scalar_count"] += 1
            self.value_diversity[scalar_text.lower()[:30]] += 1

    def _extract_all_facts(self) -> None:
        """Extract all facts and compute adaptive scores."""
        if self.payload is None:
            return

        facts_raw: list[tuple[str, str, int]] = []
        self._walk(self.payload, (), facts_raw, 0)

        total_scalars = self.structure_stats["scalar_count"]
        max_depth = self.structure_stats["max_depth"]
        diversity_factor = len(self.value_diversity) / max(1, total_scalars)

        for label, value, depth in facts_raw:
            information_value = self._compute_information_value(value, diversity_factor)
            structural_importance = self._compute_structural_importance(label, depth, max_depth)
            adaptive_score = information_value * structural_importance
            tokens = max(1, _token_count(label) + _token_count(value))
            self.all_facts.append(
                AdaptiveFact(
                    label=label,
                    value=value,
                    information_value=information_value,
                    structural_importance=structural_importance,
                    adaptive_score=adaptive_score,
                    tokens=tokens,
                    depth=depth,
                )
            )

    def _walk(self, value: Any, path: tuple[str, ...], sink: list[tuple[str, str, int]], depth: int) -> None:
        """Recursively walk JSON structure."""
        if depth > MAX_DEPTH:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                self._walk(child, path + (key,), sink, depth + 1)
            return

        if isinstance(value, list):
            for idx, child in enumerate(value[:6]):
                self._walk(child, path + (f"[{idx}]",), sink, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if not scalar_text:
            return
        label = _format_path(path)
        sink.append((label, scalar_text, depth))

    def _compute_information_value(self, value: str, diversity_factor: float) -> float:
        """Compute information value based on content characteristics."""
        if not value:
            return 0.0
        tokens = value.split()
        unique_ratio = len(set(token.lower() for token in tokens)) / max(1, len(tokens))
        has_digit = any(ch.isdigit() for ch in value)
        has_alpha = any(ch.isalpha() for ch in value)
        mixed_bonus = 1.2 if (has_digit and has_alpha) else 1.0
        length_factor = min(1.0, len(value) / 50.0)
        rarity = 1.0 / max(1, self.value_diversity.get(value.lower()[:30], 1))
        return (unique_ratio * 2.0 + mixed_bonus + length_factor + rarity) * diversity_factor

    def _compute_structural_importance(self, label: str, depth: int, max_depth: int) -> float:
        """Compute structural importance based on position in hierarchy."""
        depth_normalized = 1.0 - (depth / max(1, max_depth + 1))
        label_tokens = label.split(".")
        top_level_bonus = 1.3 if len(label_tokens) <= 2 else 1.0
        key_importance = 1.2 if any(keyword in label.lower() for keyword in ["name", "title", "id", "key"]) else 1.0
        return depth_normalized * top_level_bonus * key_importance

    def compute_adaptive_budget(self, base_budget: int) -> int:
        """Compute adaptive budget based on structure analysis."""
        total_keys = self.structure_stats["total_keys"]
        scalar_count = self.structure_stats["scalar_count"]
        max_depth = self.structure_stats["max_depth"]

        complexity_factor = math.log1p(total_keys) / 10.0
        density_factor = scalar_count / max(1, total_keys)
        depth_factor = 1.0 + (max_depth / 10.0)

        adaptive_multiplier = 1.0 + complexity_factor * density_factor * depth_factor
        return int(base_budget * adaptive_multiplier)

    def select_adaptive_facts(self, budget: int) -> list[AdaptiveFact]:
        """Select facts using adaptive scoring within budget."""
        sorted_facts = sorted(self.all_facts, key=lambda f: (-f.adaptive_score, f.tokens, f.depth))
        selected: list[AdaptiveFact] = []
        tokens_used = 0
        seen_scopes: set[str] = set()

        for fact in sorted_facts:
            scope = fact.label.split(".")[0] if "." in fact.label else fact.label
            if scope in seen_scopes and len(seen_scopes) > 5:
                continue
            if tokens_used + fact.tokens > budget:
                continue
            selected.append(fact)
            tokens_used += fact.tokens
            seen_scopes.add(scope)
            if len(selected) >= 22:
                break

        return selected


def run_iteration_adaptive_budget(task_override: str | None = None) -> AgentRunSummary:
    """Main entrypoint for adaptive budget allocation."""
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

    allocator = AdaptiveBudgetAllocator(raw_text, payload)
    word_count = _token_count(raw_text)
    base_budget = _compute_base_budget(word_count)
    adaptive_budget = allocator.compute_adaptive_budget(base_budget)
    selected_facts = allocator.select_adaptive_facts(adaptive_budget)

    summary_lines = [f"## AdaptiveBudget Summary of `{full_path.name}`"]
    summary_lines.append(
        f"Adaptive selection ({len(selected_facts)} facts, budget: {base_budget}→{adaptive_budget})"
    )

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
        "base_budget": base_budget,
        "adaptive_budget": adaptive_budget,
        "structure_stats": allocator.structure_stats,
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
        return compact[:170]
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


def _compute_base_budget(word_count: int) -> int:
    """Compute base token budget."""
    if word_count < 100:
        return 60
    if word_count < 1000:
        return int(math.sqrt(word_count) * 3.5)
    return int(math.log(word_count) * 20)


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
    summary = run_iteration_adaptive_budget()
    print(summary.model_dump())

