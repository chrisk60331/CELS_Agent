"""Benchmark runner that mimics AgentPrune-style communication pruning."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Sequence

from strands import Agent
from strands_tools import editor, file_read, file_write

from src.agent_run_summary import AgentRunSummary
from benchmarks.main import FILE_SYSTEM_PROMPT, _extract_response_text
from constants import MODEL_ID, task

DEFAULT_KEEP_RATIO = 0.65


def run_agentprune_agent(task_override: str | None = None) -> AgentRunSummary:
    """Run the default file agent with AgentPrune-inspired prompt pruning."""
    goal = (task_override or task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; populate constants.task.")

    reducer = AgentPrunePromptReducer(keep_ratio=DEFAULT_KEEP_RATIO)

    pruned_system_prompt, system_stats = reducer.prune_block(
        FILE_SYSTEM_PROMPT, force_keywords=("file", "read", "write", "modify")
    )
    pruned_goal, goal_stats = reducer.prune_block(
        goal, force_keywords=("data/", "file", "json", "summarize")
    )

    if not pruned_system_prompt:
        pruned_system_prompt = FILE_SYSTEM_PROMPT
    if not pruned_goal:
        pruned_goal = goal

    file_agent = Agent(
        system_prompt=pruned_system_prompt,
        tools=[file_read, file_write, editor],
        model=MODEL_ID,
    )

    response = file_agent(pruned_goal)
    final_answer = _extract_response_text(response)

    metrics_obj = getattr(response, "metrics", None)
    accumulated = getattr(metrics_obj, "accumulated_usage", None) if metrics_obj is not None else None

    metadata = {
        "final_answer": final_answer,
        "agentprune": {
            "system_prompt": system_stats,
            "goal": goal_stats,
            "keep_ratio": reducer.keep_ratio,
        },
    }
    return AgentRunSummary.from_usage(accumulated, metadata=metadata)


@dataclass
class AgentPrunePromptReducer:
    """Heuristic reducer inspired by https://github.com/yanweiyue/AgentPrune."""

    keep_ratio: float = DEFAULT_KEEP_RATIO

    def __post_init__(self) -> None:
        if not (0 < self.keep_ratio <= 1):
            raise ValueError("keep_ratio must be 0 < ratio <= 1.")

    def prune_block(self, text: str, force_keywords: Sequence[str]) -> tuple[str, Dict[str, int]]:
        cleaned = text.strip()
        if not cleaned:
            return "", {"original_sentences": 0, "kept_sentences": 0, "removed_sentences": 0}

        sentences = _split_sentences(cleaned)
        if not sentences:
            return cleaned, {"original_sentences": 0, "kept_sentences": 0, "removed_sentences": 0}
        if len(sentences) == 1:
            return sentences[0], {"original_sentences": 1, "kept_sentences": 1, "removed_sentences": 0}

        keep_count = max(1, int(round(len(sentences) * self.keep_ratio)))
        scored = []
        for idx, sentence in enumerate(sentences):
            tokens = re.findall(r"[A-Za-z0-9_/.-]+", sentence.lower())
            unique_ratio = len(set(tokens)) / max(1, len(tokens))
            keyword_bonus = 1.0 if any(keyword.lower() in sentence.lower() for keyword in force_keywords) else 0.0
            numeric_bonus = 0.15 if any(char.isdigit() for char in sentence) else 0.0
            score = unique_ratio + keyword_bonus + numeric_bonus
            scored.append((score, idx, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        kept = sorted(scored[:keep_count], key=lambda item: item[1])
        kept_sentences = [sentence for _, _, sentence in kept]
        pruned_text = " ".join(kept_sentences)

        stats = {
            "original_sentences": len(sentences),
            "kept_sentences": len(kept_sentences),
            "removed_sentences": len(sentences) - len(kept_sentences),
        }
        return pruned_text, stats


def _split_sentences(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    if chunks:
        return chunks
    return [text]


if __name__ == "__main__":  # pragma: no cover - manual smoke run helper
    summary = run_agentprune_agent()
    print(summary.usage or None)


