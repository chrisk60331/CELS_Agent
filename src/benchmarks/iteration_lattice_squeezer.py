"""Lattice Squeezer iteration maximizing F1-per-token via Bedrock."""
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
    "but",
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
    "this",
    "these",
    "those",
    "will",
    "shall",
    "should",
    "would",
    "can",
    "could",
    "may",
    "might",
    "about",
    "there",
    "here",
    "than",
    "then",
    "after",
    "before",
    "between",
    "while",
    "where",
    "when",
    "which",
}
MAX_JSON_DEPTH = 10
MAX_JSON_BRANCH = 80
MAX_LIST_SAMPLE = 6
MAX_SHARDS = 280
MIN_SENTENCE_TOKENS = 5
MAX_SENTENCE_TOKENS = 80
SYSTEM_PROMPT = (
    "You are Lattice Squeezer, an information compressor obsessed with factual coverage "
    "per token. Only use the supplied evidence. Preserve numeric fidelity, named entities, "
    "and chronology. Respond with compact bullet sentences that remain readable."
)


class DocumentSnapshot(BaseModel):
    """View over the original benchmark document."""

    goal: str = Field(..., description="Benchmark instruction text.")
    path: str = Field(..., description="Relative path resolved from the goal.")
    raw_text: str = Field(..., description="Exact file contents.")
    payload: Any | None = Field(None, description="Parsed JSON payload when available.")
    word_count: int = Field(..., ge=0, description="Whitespace token count of the file.")
    char_count: int = Field(..., ge=0, description="Character length of the file.")
    format_hint: str = Field(..., description="Descriptor for downstream formatting.")
    keywords: tuple[str, ...] = Field(..., description="High-priority content terms.")


class EvidenceShard(BaseModel):
    """Atomic evidence unit competing for context space."""

    text: str = Field(..., description="Renderable evidence line.")
    tokens: int = Field(..., ge=1, description="Estimated token cost.")
    score: float = Field(..., ge=0.0, description="Information weight.")
    channel: str = Field(..., description="Origin channel label.")
    topic_key: str = Field(..., description="Topic bucket for coverage control.")


class PromptBundle(BaseModel):
    """Prompt plus bookkeeping for context tokens."""

    prompt: str = Field(..., description="Full prompt sent to the LLM.")
    token_estimate: int = Field(..., ge=1, description="Estimated token count for prompt.")


def run_iteration_lattice_squeezer(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint executed by the benchmark harness."""
    goal_text = (task_override or constants.task).strip()
    if not goal_text:
        raise ValueError("Benchmark task must be non-empty.")

    snapshot = _load_snapshot(goal_text)
    shards = _build_shards(snapshot)
    if not shards:
        raise ValueError("No evidence shards generated from document.")

    context_budget = _context_budget(snapshot.word_count)
    selector = ShardSelector(context_budget=context_budget)
    selected = selector.select(shards, snapshot.keywords)
    if not selected:
        raise ValueError("Shard selector produced empty set.")

    output_budget = _output_budget(snapshot.word_count)
    composer = PromptComposer(output_budget=output_budget)
    prompt_bundle = composer.compose(snapshot, selected)

    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt_bundle.prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.1,
    )

    usage = response.usage.model_dump()
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": snapshot.path,
        "evidence_items": len(selected),
        "context_tokens_estimate": prompt_bundle.token_estimate,
        "context_budget": context_budget,
        "output_budget": output_budget,
        "keywords": snapshot.keywords,
        "channels": selector.channel_counts(selected),
    }
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


def _load_snapshot(goal_text: str) -> DocumentSnapshot:
    file_path = _extract_goal_file_path(goal_text)
    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Source file not found: {full_path}")
    raw_text = full_path.read_text(encoding="utf-8")
    payload = _try_parse_json(raw_text)
    keyword_source = raw_text if payload is None else json.dumps(payload, ensure_ascii=False)
    keywords = _extract_keywords(keyword_source)
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
        keywords=keywords,
    )


def _build_shards(snapshot: DocumentSnapshot) -> list[EvidenceShard]:
    extractor: _ShardExtractor
    if snapshot.payload is not None:
        extractor = JsonShardExtractor(snapshot.payload)
    else:
        extractor = TextShardExtractor(snapshot.raw_text)
    return extractor.extract()


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


def _extract_keywords(text: str, limit: int = 6) -> tuple[str, ...]:
    tokens = [
        token.lower()
        for token in TOKEN_PATTERN.findall(text.lower())
        if token.lower() not in STOPWORDS and len(token) > 3
    ]
    if not tokens:
        return ()
    counts = Counter(tokens)
    ordered = [word for word, _ in counts.most_common(limit * 2)]
    unique: list[str] = []
    for word in ordered:
        if word not in unique:
            unique.append(word)
        if len(unique) == limit:
            break
    return tuple(unique)


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 200
    base = math.log(word_count + 20, 1.7) * 95
    return int(max(200, min(480, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 52
    base = math.sqrt(word_count) * 1.9
    return int(max(48, min(90, base)))


def _token_count(text: str) -> int:
    return max(0, len([token for token in text.strip().split() if token]))


class _ShardExtractor:
    """Interface for shard extractors."""

    def extract(self) -> list[EvidenceShard]:
        raise NotImplementedError


class JsonShardExtractor(_ShardExtractor):
    """Extracts dense shards from JSON payloads."""

    def __init__(self, payload: Any):
        self._payload = payload
        self._shards: list[EvidenceShard] = []

    def extract(self) -> list[EvidenceShard]:
        self._walk(self._payload, (), 0)
        return self._shards[:MAX_SHARDS]

    def _walk(self, value: Any, path: tuple[str, ...], depth: int) -> None:
        if len(self._shards) >= MAX_SHARDS:
            return
        if depth > MAX_JSON_DEPTH:
            return

        if isinstance(value, dict):
            keys = list(value.keys())[:MAX_JSON_BRANCH]
            self._emit_structure(path, len(value), keys)
            for key in keys:
                self._walk(value[key], path + (key,), depth + 1)
            return

        if isinstance(value, list):
            size = len(value)
            self._emit_list(path, size)
            for idx, item in enumerate(value[:MAX_LIST_SAMPLE]):
                self._walk(item, path + (f"[{idx}]",), depth + 1)
            return

        self._emit_scalar(path, value, depth)

    def _emit_structure(self, path: tuple[str, ...], size: int, keys: Sequence[str]) -> None:
        if len(path) > 2:
            return
        label = _path_label(path) or "root"
        preview = ", ".join(keys[:5])
        suffix = "..." if len(keys) > 5 else ""
        text = f"{label} object holds {size} keys ({preview}{suffix})."
        score = 2.6 + min(1.2, len(keys) * 0.03)
        self._append(text, score, "structure", label)

    def _emit_list(self, path: tuple[str, ...], size: int) -> None:
        if len(path) > 2:
            return
        label = _path_label(path) or "root"
        text = f"{label} list spans {size} entries."
        score = 2.2 + min(1.0, math.log(size + 1, 3))
        self._append(text, score, "structure", label)

    def _emit_scalar(self, path: tuple[str, ...], value: Any, depth: int) -> None:
        formatted = _format_scalar(value)
        if not formatted:
            return
        label = _path_label(path) or "value"
        ending = "" if formatted.endswith((".", "!", "?")) else "."
        text = f"{label}: {formatted}{ending}"
        digit_bonus = 0.9 if any(ch.isdigit() for ch in formatted) else 0.3
        depth_penalty = 1.0 + depth * 0.12
        score = (3.1 + digit_bonus + min(1.0, len(formatted) / 64.0)) / depth_penalty
        self._append(text, score, "fact", label.split(".", 1)[0] if "." in label else label)

    def _append(self, text: str, score: float, channel: str, topic_key: str) -> None:
        tokens = _token_count(text)
        if tokens <= 0:
            return
        self._shards.append(
            EvidenceShard(text=text, tokens=tokens, score=score, channel=channel, topic_key=topic_key)
        )


class TextShardExtractor(_ShardExtractor):
    """Extracts salient sentences from text documents."""

    def __init__(self, raw_text: str):
        self._raw_text = raw_text

    def extract(self) -> list[EvidenceShard]:
        sentences = _split_sentences(self._raw_text)
        if not sentences:
            return []
        token_counts = Counter(_normalize_token(token) for token in TOKEN_PATTERN.findall(self._raw_text.lower()))
        shards: list[EvidenceShard] = []
        total_sentences = len(sentences)
        for idx, sentence in enumerate(sentences):
            normalized = sentence.strip()
            tokens = _token_count(normalized)
            if tokens < MIN_SENTENCE_TOKENS or tokens > MAX_SENTENCE_TOKENS:
                continue
            tf_bonus = _tf_bonus(normalized, token_counts)
            numeric_bonus = 0.8 if any(ch.isdigit() for ch in normalized) else 0.2
            position_penalty = 1.0 + (idx / max(1, total_sentences)) * 0.6
            score = (3.4 + tf_bonus + numeric_bonus) / position_penalty
            topic_bucket = f"chunk_{min(9, int(idx / max(1, total_sentences / 10)))}"
            shards.append(
                EvidenceShard(
                    text=_tighten_sentence(normalized),
                    tokens=tokens,
                    score=score,
                    channel="sentence",
                    topic_key=topic_bucket,
                )
            )
        return shards[:MAX_SHARDS]


class ShardSelector:
    """Greedy selector honoring context budget and keyword coverage."""

    def __init__(self, context_budget: int):
        self._context_budget = context_budget

    def select(self, shards: Sequence[EvidenceShard], keywords: Sequence[str]) -> list[EvidenceShard]:
        ordered = sorted(
            shards,
            key=lambda shard: (-shard.score / max(1, shard.tokens), -shard.score, shard.tokens),
        )
        selected: list[EvidenceShard] = []
        tokens_used = 0
        seen_text: set[str] = set()

        for keyword in keywords:
            candidate = self._first_matching(keyword, ordered, seen_text)
            if candidate is None:
                continue
            if tokens_used + candidate.tokens > self._context_budget:
                raise ValueError("Keyword coverage exceeds context budget.")
            selected.append(candidate)
            seen_text.add(candidate.text.lower())
            tokens_used += candidate.tokens

        for shard in ordered:
            if shard.text.lower() in seen_text:
                continue
            if tokens_used + shard.tokens > self._context_budget:
                continue
            selected.append(shard)
            seen_text.add(shard.text.lower())
            tokens_used += shard.tokens
        return selected

    def channel_counts(self, shards: Sequence[EvidenceShard]) -> dict[str, int]:
        counts: Counter[str] = Counter(shard.channel for shard in shards)
        return dict(counts)

    def _first_matching(
        self,
        keyword: str,
        ordered: Sequence[EvidenceShard],
        seen_text: set[str],
    ) -> EvidenceShard | None:
        lowered = keyword.lower()
        for shard in ordered:
            if shard.text.lower() in seen_text:
                continue
            if lowered in shard.text.lower():
                return shard
        return None


class PromptComposer:
    """Builds the final prompt with explicit compression constraints."""

    def __init__(self, output_budget: int):
        self._output_budget = output_budget

    def compose(self, snapshot: DocumentSnapshot, shards: Sequence[EvidenceShard]) -> PromptBundle:
        header_lines = [
            f"Goal: {snapshot.goal}",
            f"Source file: {Path(snapshot.path).name}",
            f"Document stats: words={snapshot.word_count} chars={snapshot.char_count} format={snapshot.format_hint}",
        ]
        if snapshot.keywords:
            header_lines.append(f"Priority terms: {', '.join(snapshot.keywords)}")

        instruction_lines = [
            "Instruction: produce the most information-dense bullet summary possible.",
            f"Output tokens must stay <= {self._output_budget} while maximizing factual recall.",
            "Respond with '-' bullet lines only and include numbers, names, and causal links without commentary.",
        ]

        evidence_lines = [f"[{shard.channel}] {shard.text}" for shard in shards]

        prompt_lines = header_lines + instruction_lines + ["Evidence:"] + evidence_lines
        prompt = "\n".join(line.strip() for line in prompt_lines if line.strip())
        token_estimate = sum(_token_count(line) for line in prompt_lines if line)
        return PromptBundle(prompt=prompt, token_estimate=token_estimate)


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _tf_bonus(sentence: str, token_counts: Counter[str]) -> float:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(sentence)]
    unique_tokens = set(token for token in tokens if token not in STOPWORDS)
    if not unique_tokens:
        return 0.2
    bonus = 0.0
    for token in unique_tokens:
        freq = token_counts.get(token, 1)
        bonus += 1.5 / (1.0 + math.log(freq + 1.0))
    return bonus / max(1, len(unique_tokens))


def _path_label(path: Sequence[str]) -> str:
    if not path:
        return "root"
    return ".".join(path)


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().split())[:240]
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _normalize_token(token: str) -> str:
    return token.lower().strip()


def _tighten_sentence(sentence: str) -> str:
    return " ".join(sentence.split())


if __name__ == "__main__":
    summary = run_iteration_lattice_squeezer()
    print(summary.model_dump())


