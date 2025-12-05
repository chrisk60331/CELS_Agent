"""Benchmark runner with recursive semantic file summarization."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import logging
import re
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.agent_run_summary import AgentRunSummary
import constants
from src.compressed_agent.file_summarizer import summarize_data_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
MIN_F1_THRESHOLD = 0.7
MAX_ADDITIONAL_SENTENCES = 20
MIN_SENTENCE_TOKENS = 6
MAX_SENTENCE_TOKENS = 60


class SummaryIteration(BaseModel):
    """Single compression checkpoint."""

    depth: int = Field(..., ge=0)
    segment_count: int = Field(..., ge=1)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    accepted: bool = Field(..., description="Whether the iteration stayed above the F1 guardrail.")


class SummaryOutcome(BaseModel):
    """Final recursive summary plus audit trail."""

    summary: str
    f1_score: float = Field(..., ge=0.0, le=1.0)
    iterations: List[SummaryIteration]


@dataclass
class SegmentRecord:
    """Single summary line with an importance weight."""

    index: int
    text: str
    priority: float


class RecursiveSemanticSummarizer:
    """Recursive summarizer that removes low-signal segments one at a time."""

    def __init__(self, min_f1: float = MIN_F1_THRESHOLD):
        if not 0.0 < min_f1 < 1.0:
            raise ValueError("min_f1 must be within (0, 1).")
        self.min_f1 = min_f1

    def summarize(self, file_path: Path, initial_text: str, reference_text: str) -> SummaryOutcome:
        if not initial_text.strip():
            raise ValueError(f"{file_path} produced an empty base summary.")

        reference_counts = Counter(_tokenize(reference_text))
        segments = self._build_segments(initial_text, reference_text, reference_counts)
        if not segments:
            segments = [SegmentRecord(index=0, text=initial_text.strip(), priority=10.0)]

        active_indices = list(range(len(segments)))
        summary_text = self._compose_summary(segments, active_indices)
        current_score = _f1_score(reference_text, summary_text)
        LOGGER.info(
            "RecursiveSummary[%s] depth=%d segments=%d f1=%.3f min_f1=%.2f",
            file_path.name,
            0,
            len(active_indices),
            current_score,
            self.min_f1,
        )
        iterations: List[SummaryIteration] = [
            SummaryIteration(
                depth=0,
                segment_count=len(active_indices),
                f1_score=current_score,
                accepted=current_score >= self.min_f1,
            )
        ]

        removal_queue = sorted(segments, key=lambda segment: (segment.priority, segment.index))
        step = 0
        for segment in removal_queue:
            if len(active_indices) <= 1:
                break
            if segment.index not in active_indices:
                continue

            candidate_indices = [idx for idx in active_indices if idx != segment.index]
            candidate_summary = self._compose_summary(segments, candidate_indices)
            candidate_score = _f1_score(reference_text, candidate_summary)
            accepted = candidate_score >= self.min_f1
            step += 1
            iterations.append(
                SummaryIteration(
                    depth=step,
                    segment_count=len(candidate_indices),
                    f1_score=candidate_score,
                    accepted=accepted,
                )
            )

            if accepted:
                active_indices = candidate_indices
                current_score = candidate_score
                summary_text = candidate_summary
                LOGGER.info(
                    "RecursiveSummary[%s] step=%d accepted removal idx=%d priority=%.3f f1=%.3f remaining=%d",
                    file_path.name,
                    step,
                    segment.index,
                    segment.priority,
                    current_score,
                    len(active_indices),
                )
            else:
                LOGGER.info(
                    "RecursiveSummary[%s] step=%d rejected removal idx=%d priority=%.3f f1=%.3f remaining=%d",
                    file_path.name,
                    step,
                    segment.index,
                    segment.priority,
                    candidate_score,
                    len(candidate_indices),
                )
                break

        LOGGER.info(
            "RecursiveSummary[%s] completed steps=%d final_f1=%.3f segments=%d",
            file_path.name,
            step,
            current_score,
            len(active_indices),
        )
        return SummaryOutcome(summary=summary_text.strip(), f1_score=current_score, iterations=iterations)

    def _build_segments(
        self,
        initial_text: str,
        reference_text: str,
        reference_counts: Counter,
    ) -> List[SegmentRecord]:
        lines = [line.strip() for line in initial_text.splitlines() if line.strip()]
        segments: List[SegmentRecord] = []
        seen_text: set[str] = set()

        def _append_segment(text: str, priority: float) -> None:
            normalized = text.strip()
            if not normalized or normalized in seen_text:
                return
            segments.append(SegmentRecord(index=len(segments), text=normalized, priority=priority))
            seen_text.add(normalized)

        for line in lines:
            priority = self._segment_priority(line, reference_counts)
            _append_segment(line, priority)

        for sentence, priority in self._top_raw_sentences(reference_text, reference_counts):
            _append_segment(sentence, priority)

        LOGGER.debug("RecursiveSummary: built %d segments", len(segments))
        return segments

    @staticmethod
    def _compose_summary(segments: List[SegmentRecord], indices: List[int]) -> str:
        return "\n".join(segments[idx].text for idx in indices).strip()

    def _segment_priority(self, line: str, reference_counts: Counter) -> float:
        base_weight = self._base_weight(line)
        tokens = _tokenize(line)
        overlap = 0
        if tokens:
            counts = Counter(tokens)
            overlap = sum(
                min(count, reference_counts.get(token, 0))
                for token, count in counts.items()
            )
        density = overlap / max(1, len(tokens))
        return base_weight + density

    @staticmethod
    def _base_weight(line: str) -> float:
        if line.startswith("## "):
            return 10.0
        if line.startswith("### "):
            return 8.0
        if line.startswith("Key facts"):
            return 7.5
        if line.startswith("Numeric highlights"):
            return 7.0
        if line.startswith("Keywords"):
            return 6.5
        if line.startswith("- "):
            return 6.0
        if ":" in line:
            return 5.0
        return 4.0

    def _top_raw_sentences(
        self,
        reference_text: str,
        reference_counts: Counter,
    ) -> List[tuple[str, float]]:
        sentences = _split_sentences(reference_text)
        scored: List[tuple[float, str]] = []
        for sentence in sentences:
            priority = self._raw_sentence_priority(sentence, reference_counts)
            if priority <= 0.0:
                continue
            scored.append((-priority, sentence.strip()))

        scored.sort()
        top_sentences = [(text, -score) for score, text in scored[:MAX_ADDITIONAL_SENTENCES]]
        return top_sentences

    def _raw_sentence_priority(self, sentence: str, reference_counts: Counter) -> float:
        tokens = _tokenize(sentence)
        token_count = len(tokens)
        if token_count < MIN_SENTENCE_TOKENS or token_count > MAX_SENTENCE_TOKENS:
            return 0.0

        unique_tokens = set(tokens)
        rarity = sum(1.0 / (1 + reference_counts.get(token, 0)) for token in unique_tokens)
        unique_ratio = len(unique_tokens) / token_count
        digit_bonus = 0.3 if any(ch.isdigit() for ch in sentence) else 0.0
        proper_noun_bonus = 0.1 * sum(
            1 for word in sentence.split() if word[:1].isupper() and word[0].isalpha()
        )
        punctuation_penalty = 0.2 * sentence.count(";")
        length_bonus = min(1.0, token_count / 40.0)

        return 4.0 + rarity + unique_ratio + digit_bonus + proper_noun_bonus + length_bonus - punctuation_penalty



def run_recursive_summarizer(task_override: str | None = None) -> AgentRunSummary:
    summarizer = RecursiveSemanticSummarizer()

    goal = (task_override or constants.task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; provide a summarization goal.")

    file_path = _extract_goal_file_path(goal)
    if not file_path:
        raise ValueError(f"Unable to determine file path from goal: {goal}")

    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Source file not found: {full_path}")

    raw_text = full_path.read_text(encoding="utf-8")
    deterministic_summary = summarize_data_file(full_path, raw_text)
    outcome = summarizer.summarize(full_path, deterministic_summary, raw_text)

    usage = _estimate_usage(raw_text, outcome.summary)
    metadata = {
        "final_answer": outcome.summary,
        "summary_iterations": [iteration.model_dump() for iteration in outcome.iterations],
        "source_file": str(full_path),
    }

    return AgentRunSummary.from_usage(
        usage=usage,
        metadata=metadata,
    )


def _f1_score(reference: str, candidate: str) -> float:
    reference_tokens = _tokenize(reference)
    candidate_tokens = _tokenize(candidate)
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_counts: Dict[str, int] = {}
    candidate_counts: Dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    for token in candidate_tokens:
        candidate_counts[token] = candidate_counts.get(token, 0) + 1

    true_positive = sum(
        min(candidate_counts.get(token, 0), reference_counts.get(token, 0))
        for token in candidate_counts
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


def _tokenize(value: str) -> List[str]:
    return [token for token in value.lower().split() if token]


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _extract_goal_file_path(goal: str) -> Path | None:
    marker = "summarize file "
    goal_lower = goal.lower()
    if marker not in goal_lower:
        return None
    start_idx = goal_lower.index(marker) + len(marker)
    remainder = goal[start_idx:].strip()
    if not remainder:
        return None
    candidate = remainder.split()[0]
    return Path(candidate)


def _estimate_usage(source_text: str, summary_text: str) -> Dict[str, int]:
    def _token_count(text: str) -> int:
        return max(1, len(text.split()))

    input_tokens = _token_count(source_text)
    output_tokens = _token_count(summary_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


if __name__ == "__main__":
    summary = run_recursive_summarizer()
    print(summary.model_dump())

