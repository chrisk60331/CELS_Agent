"""Spectral Flux iteration enforcing F1-per-token optimality with Bedrock."""
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
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "with",
}
MAX_SLICE_DEPTH = 10
MAX_BRANCH = 80
MAX_LIST_SAMPLE = 8
MAX_SLICES = 340
MIN_SENTENCE_TOKENS = 6
MAX_SENTENCE_TOKENS = 72
SYSTEM_PROMPT = (
    "You are Spectral Flux, an information compressor that maximizes benchmark F1 per token. "
    "Use only the evidence provided, output dense bullet sentences, and never speculate."
)


class DocumentSnapshot(BaseModel):
    """Normalized view of the benchmark document."""

    goal: str = Field(..., description="Benchmark instruction text.")
    path: str = Field(..., description="Relative source path parsed from the goal.")
    raw_text: str = Field(..., description="Exact file contents.")
    payload: Any | None = Field(None, description="Parsed JSON payload when available.")
    word_count: int = Field(..., ge=0, description="Whitespace token count for raw text.")
    char_count: int = Field(..., ge=0, description="Total character length.")
    format_hint: str = Field(..., description="Human readable descriptor for format.")


class EvidenceSlice(BaseModel):
    """Atomic evidence competing for prompt context."""

    text: str = Field(..., description="Renderable evidence line.")
    tokens: int = Field(..., ge=1, description="Estimated token cost.")
    score: float = Field(..., ge=0.0, description="Information weight.")
    channel: str = Field(..., description="Origin channel.")
    topic: str = Field(..., description="Topic bucket for coverage constraints.")


class PromptPackage(BaseModel):
    """Prompt plus bookkeeping for auditing budgets."""

    prompt: str = Field(..., description="Prompt sent to Bedrock.")
    prompt_tokens: int = Field(..., ge=1, description="Estimated prompt token count.")


def run_iteration_spectral_flux(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint executed by the benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    snapshot = _load_snapshot(goal)

    builder = JsonSliceBuilder(snapshot.payload) if snapshot.payload is not None else TextSliceBuilder(snapshot.raw_text)
    slices = builder.build()
    if not slices:
        raise ValueError("Spectral Flux: slice builder returned no evidence.")

    context_budget = _context_budget(snapshot.word_count)
    selector = SliceSelector(context_budget=context_budget)
    selected = selector.pick(slices)
    if not selected:
        raise ValueError("Spectral Flux: selector did not admit any slices.")

    output_budget = _output_budget(snapshot.word_count)
    prompt_package = PromptBuilder(output_budget=output_budget).build(snapshot, selected)

    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt_package.prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.15,
    )

    usage = response.usage.model_dump()
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": snapshot.path,
        "context_budget": context_budget,
        "context_tokens_estimate": prompt_package.prompt_tokens,
        "output_budget": output_budget,
        "selected_slices": len(selected),
        "channels": selector.channel_counts(selected),
    }
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


def _require_goal(goal: str) -> str:
    normalized = goal.strip()
    if not normalized:
        raise ValueError("Benchmark task must be populated.")
    return normalized


def _load_snapshot(goal: str) -> DocumentSnapshot:
    file_path = _extract_goal_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Spectral Flux source file missing: {full_path}")
    raw_text = full_path.read_text(encoding="utf-8")
    payload = _try_parse_json(raw_text)
    return DocumentSnapshot(
        goal=goal,
        path=str(file_path),
        raw_text=raw_text,
        payload=payload,
        word_count=_token_count(raw_text),
        char_count=len(raw_text),
        format_hint=_format_hint(payload),
    )


def _extract_goal_file_path(goal: str) -> Path:
    marker = "file "
    lowered = goal.lower()
    if marker not in lowered:
        raise ValueError("Goal must reference 'file <path>'.")
    start_idx = lowered.index(marker) + len(marker)
    remainder = goal[start_idx:].strip()
    if not remainder:
        raise ValueError("Goal missing path after 'file'.")
    return Path(remainder.split()[0])


def _try_parse_json(raw_text: str) -> Any | None:
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


def _token_count(text: str) -> int:
    tokens = [token for token in text.strip().split() if token]
    return len(tokens)


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 220
    base = math.log(word_count + 10, 1.8) * 110
    return int(max(200, min(520, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 56
    base = math.sqrt(word_count) * 2.1
    return int(max(54, min(96, base)))


class JsonSliceBuilder:
    """Extracts structured slices from JSON payloads."""

    def __init__(self, payload: Any):
        self._payload = payload
        self._slices: list[EvidenceSlice] = []

    def build(self) -> list[EvidenceSlice]:
        self._walk(self._payload, (), 0)
        return self._slices[:MAX_SLICES]

    def _walk(self, value: Any, path: tuple[str, ...], depth: int) -> None:
        if len(self._slices) >= MAX_SLICES:
            return
        if depth > MAX_SLICE_DEPTH:
            return

        if isinstance(value, dict):
            keys = list(value.keys())[:MAX_BRANCH]
            self._emit_object(path, len(value), keys)
            for key in keys:
                self._walk(value[key], path + (key,), depth + 1)
            return

        if isinstance(value, list):
            size = len(value)
            self._emit_list(path, size)
            for idx, child in enumerate(value[:MAX_LIST_SAMPLE]):
                self._walk(child, path + (f"[{idx}]",), depth + 1)
            return

        self._emit_scalar(path, value, depth)

    def _emit_object(self, path: tuple[str, ...], size: int, keys: Sequence[str]) -> None:
        if len(path) > 2:
            return
        label = _path_label(path) or "root"
        preview = ", ".join(keys[:5])
        suffix = "..." if len(keys) > 5 else ""
        text = f"{label} object holds {size} keys ({preview}{suffix})."
        score = 2.4 + min(1.4, size * 0.02)
        self._append(text=text, score=score, channel="json.object", topic=label)

    def _emit_list(self, path: tuple[str, ...], size: int) -> None:
        if len(path) > 2:
            return
        label = _path_label(path) or "root"
        text = f"{label} list spans {size} entries."
        score = 2.1 + min(1.2, math.log(size + 1, 2.7))
        self._append(text=text, score=score, channel="json.list", topic=label)

    def _emit_scalar(self, path: tuple[str, ...], value: Any, depth: int) -> None:
        formatted = _format_scalar(value)
        if not formatted:
            return
        label = _path_label(path) or "value"
        ending = "" if formatted.endswith((".", "!", "?")) else "."
        text = f"{label}: {formatted}{ending}"
        digit_bonus = 0.9 if any(ch.isdigit() for ch in formatted) else 0.2
        depth_penalty = 1.0 + depth * 0.18
        score = (3.3 + digit_bonus + min(1.0, len(formatted) / 80.0)) / depth_penalty
        topic = label.split(".", 1)[0] if "." in label else label
        self._append(text=text, score=score, channel="json.scalar", topic=topic)

    def _append(self, text: str, score: float, channel: str, topic: str) -> None:
        tokens = _token_count(text)
        if tokens <= 0:
            return
        self._slices.append(
            EvidenceSlice(text=text, tokens=tokens, score=score, channel=channel, topic=topic or channel)
        )


class TextSliceBuilder:
    """Extracts salient spans from raw text documents."""

    def __init__(self, raw_text: str):
        self._raw_text = raw_text

    def build(self) -> list[EvidenceSlice]:
        sentences = _split_sentences(self._raw_text)
        if not sentences:
            return []
        token_counts = Counter(
            token
            for token in TOKEN_PATTERN.findall(self._raw_text.lower())
            if token and token not in STOPWORDS
        )
        slices: list[EvidenceSlice] = []
        total_sentences = len(sentences)
        for idx, sentence in enumerate(sentences):
            tightened = " ".join(sentence.split())
            tokens = _token_count(tightened)
            if tokens < MIN_SENTENCE_TOKENS or tokens > MAX_SENTENCE_TOKENS:
                continue
            unique_tokens = {
                token
                for token in TOKEN_PATTERN.findall(tightened.lower())
                if token and token not in STOPWORDS
            }
            if not unique_tokens:
                continue
            rarity = sum(1.0 / (1.0 + token_counts.get(token, 0)) for token in unique_tokens) / len(unique_tokens)
            numeric_bonus = 0.7 if any(ch.isdigit() for ch in tightened) else 0.1
            position_penalty = 1.0 + (idx / max(1, total_sentences)) * 0.5
            score = (3.2 + rarity + numeric_bonus) / position_penalty
            topic = f"segment_{min(9, idx * 10 // max(1, total_sentences))}"
            slices.append(
                EvidenceSlice(
                    text=tightened,
                    tokens=tokens,
                    score=score,
                    channel="text.sentence",
                    topic=topic,
                )
            )
            if len(slices) >= MAX_SLICES:
                break
        return slices


class SliceSelector:
    """Greedy selector enforcing coverage and budget constraints."""

    def __init__(self, context_budget: int):
        self._context_budget = context_budget

    def pick(self, slices: Sequence[EvidenceSlice]) -> list[EvidenceSlice]:
        ordered = sorted(
            slices,
            key=lambda item: (-item.score / max(1, item.tokens), -item.score, item.tokens),
        )
        selected: list[EvidenceSlice] = []
        tokens_used = 0
        topic_counts: dict[str, int] = {}
        seen_text: set[str] = set()

        for slc in ordered:
            normalized_text = slc.text.lower()
            topic_total = topic_counts.get(slc.topic, 0)
            if normalized_text in seen_text:
                continue
            if topic_total >= 5:
                continue
            if tokens_used + slc.tokens > self._context_budget:
                continue
            selected.append(slc)
            seen_text.add(normalized_text)
            topic_counts[slc.topic] = topic_total + 1
            tokens_used += slc.tokens
        return selected

    def channel_counts(self, slices: Sequence[EvidenceSlice]) -> dict[str, int]:
        counts = Counter(slc.channel for slc in slices)
        return dict(counts)


class PromptBuilder:
    """Assembles the final prompt for the Bedrock call."""

    def __init__(self, output_budget: int):
        self._output_budget = output_budget

    def build(self, snapshot: DocumentSnapshot, slices: Sequence[EvidenceSlice]) -> PromptPackage:
        header_lines = [
            f"Goal: {snapshot.goal}",
            f"Source file: {Path(snapshot.path).name}",
            f"Document stats: words={snapshot.word_count} chars={snapshot.char_count} format={snapshot.format_hint}",
        ]
        instruction_lines = [
            "Instruction: compress the evidence into max-information bullets.",
            f"Absolute output ceiling: {self._output_budget} tokens.",
            "Rules: only cite evidence, keep numbers exact, combine related facts when safe.",
        ]
        evidence_lines = [f"[{slc.channel} | {slc.topic}] {slc.text}" for slc in slices]
        prompt_lines = header_lines + instruction_lines + ["Evidence:"] + evidence_lines
        prompt = "\n".join(prompt_lines)
        prompt_tokens = sum(_token_count(line) for line in prompt_lines if line)
        return PromptPackage(prompt=prompt, prompt_tokens=prompt_tokens)


def _path_label(path: Sequence[str]) -> str:
    if not path:
        return "root"
    return ".".join(path)


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        return compact[:240]
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


if __name__ == "__main__":
    summary = run_iteration_spectral_flux()
    print(summary.model_dump())


