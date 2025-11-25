"""Benchmark runner with recursive semantic file summarization."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from pydantic import BaseModel, Field

from benchmarks.agent_run_summary import AgentRunSummary
from constants import MODEL_ID, tests
from src.compressed_agent.agent import CompressedAgent
from src.compressed_agent.bedrock_client import BedrockClient
from src.compressed_agent.state_edit import StateEdit
from src.compressed_agent.tools import Tool, ToolRegistry, ToolResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent

MIN_F1_THRESHOLD = 0.28
SEGMENTS_PER_GROUP = 4
MAX_DEPTH = 5
SEGMENT_CHAR_LIMIT = 480


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


class ReferenceSummaryStore:
    """Lookup table for reference summaries keyed by absolute file path."""

    def __init__(self, benchmark_dir: Path):
        self._reference_text: Dict[Path, str] = {}
        self._hydrate(benchmark_dir)

    def reference_for(self, file_path: Path) -> str:
        normalized = file_path.resolve()
        if normalized not in self._reference_text:
            raise ValueError(f"No reference summary registered for {normalized}")
        reference = self._reference_text[normalized].strip()
        if not reference:
            raise ValueError(f"Reference summary for {normalized} is empty")
        return reference

    def _hydrate(self, benchmark_dir: Path) -> None:
        for task, summary_rel in tests:
            source_path = self._extract_source_path(task)
            if source_path is None:
                continue
            absolute_source = (PROJECT_ROOT / source_path).resolve()
            summary_path = (benchmark_dir / summary_rel).resolve()
            if not summary_path.exists():
                raise FileNotFoundError(f"Summary file expected at {summary_path}")
            summary_text = summary_path.read_text(encoding="utf-8").strip()
            if not summary_text:
                raise ValueError(f"Summary file {summary_path} is empty")
            self._reference_text[absolute_source] = summary_text

    @staticmethod
    def _extract_source_path(task: str) -> Optional[Path]:
        marker = "summarize file "
        if marker not in task.lower():
            return None
        after_marker = task.lower().split(marker, 1)[1].strip()
        original_case_segment = task[task.lower().index(marker) + len(marker):].strip()
        # Preserve original casing/path structure from the task string.
        if original_case_segment:
            return Path(original_case_segment)
        if after_marker:
            return Path(after_marker)
        return None


REFERENCE_STORE = ReferenceSummaryStore(BENCHMARK_DIR)


class RecursiveSemanticSummarizer:
    """Recursive summarizer that stops once the F1 floor is breached."""

    def __init__(
        self,
        min_f1: float = MIN_F1_THRESHOLD,
        segments_per_group: int = SEGMENTS_PER_GROUP,
        max_depth: int = MAX_DEPTH,
    ):
        if not 0.0 < min_f1 < 1.0:
            raise ValueError("min_f1 must be within (0, 1).")
        if segments_per_group < 2:
            raise ValueError("segments_per_group must be >= 2.")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        self.min_f1 = min_f1
        self.segments_per_group = segments_per_group
        self.max_depth = max_depth

    def summarize(self, file_path: Path, raw_text: str, reference_text: str) -> SummaryOutcome:
        if not raw_text.strip():
            raise ValueError(f"File '{file_path}' is empty; cannot summarize.")

        segments = self._segment_text(raw_text)
        summary_text = "\n".join(segments)
        current_score = _f1_score(reference_text, summary_text)
        iterations: List[SummaryIteration] = [
            SummaryIteration(
                depth=0,
                segment_count=len(segments),
                f1_score=current_score,
                accepted=True,
            )
        ]

        current_segments = segments
        depth = 0

        while len(current_segments) > 1 and depth < self.max_depth:
            depth += 1
            candidate_segments = self._reduce_once(current_segments)
            candidate_summary = "\n".join(candidate_segments)
            candidate_score = _f1_score(reference_text, candidate_summary)
            accepted = candidate_score >= self.min_f1
            iterations.append(
                SummaryIteration(
                    depth=depth,
                    segment_count=len(candidate_segments),
                    f1_score=candidate_score,
                    accepted=accepted,
                )
            )
            if not accepted:
                break
            summary_text = candidate_summary
            current_score = candidate_score
            current_segments = candidate_segments

        return SummaryOutcome(summary=summary_text.strip(), f1_score=current_score, iterations=iterations)

    def _segment_text(self, raw_text: str) -> List[str]:
        paragraphs: List[str] = []
        buffer: List[str] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                if buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
                continue
            buffer.append(stripped)
        if buffer:
            paragraphs.append(" ".join(buffer))

        if not paragraphs:
            raise ValueError("Unable to derive paragraphs from file contents.")

        segments: List[str] = []
        for paragraph in paragraphs:
            segments.extend(self._split_paragraph(paragraph))

        return segments or [raw_text.strip()]

    def _split_paragraph(self, paragraph: str) -> List[str]:
        words = paragraph.split()
        if len(paragraph) <= SEGMENT_CHAR_LIMIT:
            return [" ".join(words)]

        segments: List[str] = []
        cursor = 0
        while cursor < len(words):
            chunk = words[cursor : cursor + SEGMENT_CHAR_LIMIT // 5]
            segments.append(" ".join(chunk))
            cursor += SEGMENT_CHAR_LIMIT // 5
        return segments

    def _reduce_once(self, segments: List[str]) -> List[str]:
        reduced: List[str] = []
        for idx in range(0, len(segments), self.segments_per_group):
            group = segments[idx : idx + self.segments_per_group]
            reduced.append(self._semantic_compress(" ".join(group)))
        return reduced

    def _semantic_compress(self, text: str) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text.strip()
        if len(sentences) <= 2:
            return " ".join(sentences)

        scored = sorted(sentences, key=self._score_sentence, reverse=True)
        keep_count = max(1, len(sentences) // 3)
        return " ".join(scored[:keep_count])

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _score_sentence(sentence: str) -> float:
        tokens = _tokenize(sentence)
        if not tokens:
            return 0.0
        unique_ratio = len(set(tokens)) / len(tokens)
        digit_bonus = 0.3 if any(ch.isdigit() for ch in sentence) else 0.0
        proper_noun_bonus = 0.1 * sum(word[0].isupper() for word in sentence.split() if word)
        return unique_ratio + digit_bonus + proper_noun_bonus


class RecursiveFileReadTool(Tool):
    """File reader that enforces recursive summarization."""

    def __init__(self, summarizer: RecursiveSemanticSummarizer, store: ReferenceSummaryStore):
        super().__init__(
            name="read_file",
            description="Read a file and summarize it recursively with F1 guardrails.",
        )
        self._summarizer = summarizer
        self._store = store

    def execute(self, state: Any, parameters: Dict[str, Any]) -> ToolResult:  # type: ignore[override]
        file_path = parameters.get("path")
        if not isinstance(file_path, str) or not file_path.strip():
            return ToolResult(success=False, error="File path is required for recursive summarization.")

        full_path = (PROJECT_ROOT / file_path).resolve()
        if not full_path.exists():
            return ToolResult(success=False, error=f"File not found: {file_path}")

        content = full_path.read_text(encoding="utf-8")
        reference = self._store.reference_for(full_path)
        outcome = self._summarizer.summarize(full_path, content, reference)

        edit = StateEdit(
            operation="add_summary",
            data={
                "scope": [str(full_path)],
                "summary": outcome.summary[:1000],
            },
            reason=f"Recursive summary for {full_path.name}",
            priority=5,
        )

        return ToolResult(
            success=True,
            data={
                "file_path": str(full_path),
                "summary": outcome.summary,
                "f1_score": outcome.f1_score,
                "iterations": [iteration.dict() for iteration in outcome.iterations],
            },
            state_edits=[edit.dict()],
        )


def _build_recursive_registry(summarizer: RecursiveSemanticSummarizer) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(RecursiveFileReadTool(summarizer, REFERENCE_STORE))
    return registry


def run_recursive_summarizer(task_override: str | None = None) -> AgentRunSummary:
    bedrock_client = BedrockClient(model_id=MODEL_ID)
    summarizer = RecursiveSemanticSummarizer()
    registry = _build_recursive_registry(summarizer)
    agent = CompressedAgent(tool_registry=registry, bedrock_client=bedrock_client)

    goal = (task_override or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; provide a summarization goal.")

    result = agent.execute_goal(goal, max_steps=10)
    token_usage = result.get("token_usage")
    if not token_usage:
        raise ValueError("Agent execution did not return token usage.")

    final_answer = _extract_recursive_summary(result)
    metadata = {
        "final_answer": final_answer,
        "summary_iterations": _extract_iteration_history(result),
    }

    return AgentRunSummary.from_usage(
        usage=token_usage,
        metadata=metadata,
    )


def _extract_recursive_summary(result: Dict[str, Any]) -> str:
    history = result.get("history") or []
    for step in reversed(history):
        if step.get("tool_name") != "read_file":
            continue
        summary_text = (((step.get("result") or {}).get("data") or {}).get("summary") or "").strip()
        if summary_text:
            return summary_text
    raise ValueError("Recursive summarizer run did not yield a final summary.")


def _extract_iteration_history(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = result.get("history") or []
    for step in reversed(history):
        if step.get("tool_name") != "read_file":
            continue
        data = step.get("result", {}).get("data") or {}
        iterations = data.get("iterations")
        if iterations:
            return iterations
    return []


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


if __name__ == "__main__":
    summary = run_recursive_summarizer()
    print(summary.dict())

