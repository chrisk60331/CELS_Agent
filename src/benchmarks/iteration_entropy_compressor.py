"""Information-theoretic compression using entropy-based fact selection for optimal F1/token."""
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
MAX_DEPTH = 8
MAX_CANDIDATES = 500


class EntropyFact(BaseModel):
    """Fact with computed information-theoretic metrics."""

    label: str = Field(..., description="Field path")
    value: str = Field(..., description="Scalar value")
    entropy: float = Field(..., description="Shannon entropy of value")
    mutual_info: float = Field(..., description="Mutual information with document")
    compression_ratio: float = Field(..., description="Information density (bits/token)")
    tokens: int = Field(..., ge=1, description="Token count")


class EntropyCompressor:
    """Compresses documents using information-theoretic principles."""

    def __init__(self, raw_text: str, payload: Any | None):
        self.raw_text = raw_text
        self.payload = payload
        self.all_facts: list[EntropyFact] = []
        self.value_frequency: Counter[str] = Counter()
        self.label_frequency: Counter[str] = Counter()
        self._extract_all_facts()

    def _extract_all_facts(self) -> None:
        """Extract all facts and compute entropy metrics."""
        if self.payload is None:
            return

        facts_raw: list[tuple[str, str]] = []
        self._walk(self.payload, (), facts_raw, 0)

        for label, value in facts_raw:
            self.value_frequency[value.lower()] += 1
            self.label_frequency[label.lower()] += 1

        total_facts = len(facts_raw)
        for label, value in facts_raw:
            entropy = self._compute_entropy(value)
            mutual_info = self._compute_mutual_information(label, value, total_facts)
            tokens = max(1, _token_count(label) + _token_count(value))
            compression_ratio = (entropy + mutual_info) / max(1, tokens)
            self.all_facts.append(
                EntropyFact(
                    label=label,
                    value=value,
                    entropy=entropy,
                    mutual_info=mutual_info,
                    compression_ratio=compression_ratio,
                    tokens=tokens,
                )
            )

    def _walk(self, value: Any, path: tuple[str, ...], sink: list[tuple[str, str]], depth: int) -> None:
        """Recursively walk JSON structure."""
        if len(sink) >= MAX_CANDIDATES or depth > MAX_DEPTH:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                self._walk(child, path + (key,), sink, depth + 1)
            return

        if isinstance(value, list):
            for idx, child in enumerate(value[:10]):
                self._walk(child, path + (f"[{idx}]",), sink, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if not scalar_text:
            return
        label = _format_path(path)
        sink.append((label, scalar_text))

    def _compute_entropy(self, value: str) -> float:
        """Compute Shannon entropy of value."""
        if not value:
            return 0.0
        value_lower = value.lower()
        char_freq = Counter(value_lower)
        total_chars = len(value_lower)
        if total_chars == 0:
            return 0.0
        entropy = -sum((freq / total_chars) * math.log2(freq / total_chars) for freq in char_freq.values() if freq > 0)
        return entropy

    def _compute_mutual_information(self, label: str, value: str, total_facts: int) -> float:
        """Estimate mutual information between label and value."""
        if total_facts == 0:
            return 0.0
        value_lower = value.lower()
        label_lower = label.lower()
        p_value = self.value_frequency[value_lower] / total_facts
        p_label = self.label_frequency[label_lower] / total_facts
        if p_value == 0 or p_label == 0:
            return 0.0
        p_joint = 1.0 / total_facts
        if p_joint == 0:
            return 0.0
        mi = p_joint * math.log2(p_joint / (p_value * p_label)) if p_value * p_label > 0 else 0.0
        return max(0.0, mi)

    def select_optimal_facts(self, budget: int) -> list[EntropyFact]:
        """Select facts that maximize information per token within budget."""
        sorted_facts = sorted(self.all_facts, key=lambda f: (-f.compression_ratio, f.tokens))
        selected: list[EntropyFact] = []
        tokens_used = 0
        seen_labels: set[str] = set()
        seen_values: set[str] = set()

        for fact in sorted_facts:
            label_key = fact.label.lower()
            value_key = fact.value.lower()[:50]
            if label_key in seen_labels and value_key in seen_values:
                continue
            if tokens_used + fact.tokens > budget:
                continue
            selected.append(fact)
            tokens_used += fact.tokens
            seen_labels.add(label_key)
            seen_values.add(value_key)
            if len(selected) >= 30:
                break

        return selected


def run_iteration_entropy_compressor(task_override: str | None = None) -> AgentRunSummary:
    """Main entrypoint for entropy-based compression."""
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

    compressor = EntropyCompressor(raw_text, payload)
    word_count = _token_count(raw_text)
    budget = _compute_adaptive_budget(word_count)
    selected_facts = compressor.select_optimal_facts(budget)

    summary_lines = [f"## EntropyCompressor Summary of `{full_path.name}`"]
    summary_lines.append(f"Information-optimized selection ({len(selected_facts)} facts, {budget} token budget)")

    for fact in selected_facts:
        summary_lines.append(f"- {fact.label}: {fact.value}")

    if len(selected_facts) < 5:
        summary_lines.append("\n### Additional Context")
        base_lines = _extract_summary_lines(base_summary)
        summary_lines.extend(f"- {line}" for line in base_lines[:3])

    summary_text = "\n".join(summary_lines).strip()

    usage = _estimate_usage(raw_text, summary_text)
    metadata = {
        "final_answer": summary_text,
        "source_file": str(file_path),
        "fact_count": len(selected_facts),
        "token_budget": budget,
        "avg_compression_ratio": sum(f.compression_ratio for f in selected_facts) / len(selected_facts) if selected_facts else 0.0,
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
        return compact[:200]
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


def _compute_adaptive_budget(word_count: int) -> int:
    """Compute adaptive token budget based on document size."""
    if word_count < 100:
        return 60
    if word_count < 1000:
        return int(math.sqrt(word_count) * 4)
    if word_count < 10000:
        return int(math.log(word_count) * 25)
    return int(math.log(word_count) * 30)


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
    summary = run_iteration_entropy_compressor()
    print(summary.model_dump())

