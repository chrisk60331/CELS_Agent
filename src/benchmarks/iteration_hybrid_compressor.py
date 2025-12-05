"""Hybrid compression combining entropy, semantic density, and graph centrality for optimal F1/token."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.file_summarizer import summarize_data_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_DEPTH = 7
MAX_CANDIDATES = 450


class HybridFact(BaseModel):
    """Fact with hybrid scoring combining multiple approaches."""

    label: str = Field(..., description="Field path")
    value: str = Field(..., description="Scalar value")
    entropy_score: float = Field(..., description="Information-theoretic entropy")
    density_score: float = Field(..., description="Semantic density")
    centrality_score: float = Field(..., description="Graph centrality")
    hybrid_score: float = Field(..., description="Combined hybrid score")
    tokens: int = Field(..., ge=1, description="Token count")
    depth: int = Field(..., ge=0, description="Depth in structure")


class HybridCompressor:
    """Hybrid compressor combining multiple compression strategies."""

    def __init__(self, raw_text: str, payload: Any | None):
        self.raw_text = raw_text
        self.payload = payload
        self.all_facts: list[HybridFact] = []
        self.word_frequency: Counter[str] = Counter()
        self.label_frequency: Counter[str] = Counter()
        self.value_frequency: Counter[str] = Counter()
        self.node_connections: defaultdict[str, set[str]] = defaultdict(set)
        self._analyze_document()
        self._extract_all_facts()

    def _analyze_document(self) -> None:
        """Analyze document to build frequency and connection models."""
        if self.payload is None:
            return
        self._build_models(self.payload, (), None, 0)

    def _build_models(self, value: Any, path: tuple[str, ...], parent_id: str | None, depth: int) -> None:
        """Build frequency and connection models."""
        if depth > MAX_DEPTH:
            return

        current_id = ".".join(path) if path else "root"

        if isinstance(value, dict):
            if parent_id:
                self.node_connections[parent_id].add(current_id)
                self.node_connections[current_id].add(parent_id)
            for key, child in value.items():
                self.label_frequency[key.lower()] += 1
                self._build_models(child, path + (key,), current_id, depth + 1)
            return

        if isinstance(value, list):
            if parent_id:
                self.node_connections[parent_id].add(current_id)
                self.node_connections[current_id].add(parent_id)
            for child in value[:15]:
                self._build_models(child, path + (f"[{len(value)}]",), current_id, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if scalar_text:
            if parent_id:
                self.node_connections[parent_id].add(current_id)
                self.node_connections[current_id].add(parent_id)
            words = scalar_text.lower().split()
            self.word_frequency.update(words)
            self.value_frequency[scalar_text.lower()[:40]] += 1

    def _extract_all_facts(self) -> None:
        """Extract all facts and compute hybrid scores."""
        if self.payload is None:
            return

        facts_raw: list[tuple[str, str, int]] = []
        self._walk(self.payload, (), facts_raw, 0)

        total_words = sum(self.word_frequency.values()) if self.word_frequency else 1
        total_facts = len(facts_raw)

        for label, value, depth in facts_raw:
            entropy_score = self._compute_entropy_score(value, total_facts)
            density_score = self._compute_density_score(value, total_words)
            centrality_score = self._compute_centrality_score(label, depth)
            hybrid_score = (entropy_score * 0.35 + density_score * 0.40 + centrality_score * 0.25)
            tokens = max(1, _token_count(label) + _token_count(value))
            self.all_facts.append(
                HybridFact(
                    label=label,
                    value=value,
                    entropy_score=entropy_score,
                    density_score=density_score,
                    centrality_score=centrality_score,
                    hybrid_score=hybrid_score,
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
            for idx, child in enumerate(value[:7]):
                self._walk(child, path + (f"[{idx}]",), sink, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if not scalar_text:
            return
        label = _format_path(path)
        sink.append((label, scalar_text, depth))

    def _compute_entropy_score(self, value: str, total_facts: int) -> float:
        """Compute entropy-based information score."""
        if not value or total_facts == 0:
            return 0.0
        char_freq = Counter(value.lower())
        total_chars = len(value.lower())
        if total_chars == 0:
            return 0.0
        entropy = -sum(
            (freq / total_chars) * math.log2(freq / total_chars)
            for freq in char_freq.values()
            if freq > 0
        )
        value_freq = self.value_frequency.get(value.lower()[:40], 0)
        rarity = math.log1p(total_facts / max(1, value_freq))
        return entropy * 0.6 + rarity * 0.4

    def _compute_density_score(self, value: str, total_words: int) -> float:
        """Compute semantic density score."""
        if not value or total_words == 0:
            return 0.0
        words = value.lower().split()
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        avg_frequency = sum(self.word_frequency.get(word, 0) for word in words) / len(words)
        rarity = math.log1p(total_words / max(1, avg_frequency))
        has_digit = any(ch.isdigit() for ch in value)
        has_alpha = any(ch.isalpha() for ch in value)
        mixed_bonus = 0.4 if (has_digit and has_alpha) else 0.0
        return (unique_ratio * 1.5 + rarity * 0.8 + mixed_bonus)

    def _compute_centrality_score(self, label: str, depth: int) -> float:
        """Compute graph centrality score."""
        node_id = label
        connections = len(self.node_connections.get(node_id, set()))
        depth_penalty = 1.0 / (1.0 + depth * 0.25)
        label_freq = self.label_frequency.get(label.split(".")[-1].lower() if "." in label else label.lower(), 0)
        label_uniqueness = math.log1p(100 / max(1, label_freq))
        return (connections * 0.5 + depth_penalty * 1.0 + label_uniqueness * 0.3)

    def select_hybrid_facts(self, budget: int) -> list[HybridFact]:
        """Select facts using hybrid scoring within budget."""
        sorted_facts = sorted(self.all_facts, key=lambda f: (-f.hybrid_score, f.tokens, f.depth))
        selected: list[HybridFact] = []
        tokens_used = 0
        seen_scopes: set[str] = set()

        for fact in sorted_facts:
            scope = fact.label.split(".")[0] if "." in fact.label else fact.label
            if scope in seen_scopes and len(seen_scopes) > 6:
                continue
            if tokens_used + fact.tokens > budget:
                continue
            selected.append(fact)
            tokens_used += fact.tokens
            seen_scopes.add(scope)
            if len(selected) >= 24:
                break

        return selected


def run_iteration_hybrid_compressor(task_override: str | None = None) -> AgentRunSummary:
    """Main entrypoint for hybrid compression."""
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

    compressor = HybridCompressor(raw_text, payload)
    word_count = _token_count(raw_text)
    budget = _compute_hybrid_budget(word_count)
    selected_facts = compressor.select_hybrid_facts(budget)

    summary_lines = [f"## HybridCompressor Summary of `{full_path.name}`"]
    avg_score = sum(f.hybrid_score for f in selected_facts) / len(selected_facts) if selected_facts else 0.0
    summary_lines.append(f"Hybrid selection ({len(selected_facts)} facts, avg score: {avg_score:.3f})")

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
        "avg_hybrid_score": avg_score,
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
        return compact[:175]
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


def _compute_hybrid_budget(word_count: int) -> int:
    """Compute token budget for hybrid compression."""
    if word_count < 150:
        return 65
    if word_count < 1500:
        return int(math.pow(word_count, 0.42) * 7)
    return int(math.log(word_count) * 21)


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
    summary = run_iteration_hybrid_compressor()
    print(summary.model_dump())

