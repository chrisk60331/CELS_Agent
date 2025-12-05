"""Temporal synthesis benchmark using Bedrock for historical summaries."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATE_PATTERN = re.compile(r"\b(1[0-2][0-9]{2}|[0-9]{4})\b")
TERM_PATTERN = re.compile(r"[a-z0-9]+")

REGION_KEYWORDS = {
    "europe": "Europe",
    "asia": "Asia",
    "africa": "Africa",
    "americas": "Americas",
    "america": "Americas",
    "oceania": "Oceania",
    "middle east": "Middle East",
}

THEME_KEYWORDS = {
    "succession": "Political change",
    "crown": "Political change",
    "king": "Political change",
    "emperor": "Political change",
    "battle": "Military conflict",
    "siege": "Military conflict",
    "war": "Military conflict",
    "crusade": "Military conflict",
    "church": "Religious events",
    "papal": "Religious events",
    "patriarch": "Religious events",
    "monastery": "Religious events",
    "cathedral": "Cultural achievements",
    "invention": "Technology",
    "printing": "Technology",
    "astronomy": "Scientific discovery",
}


class RankedSlice(BaseModel):
    """Snippet plus heuristic score for downstream synthesis."""

    text: str
    score: float
    token_count: int
    earliest_year: int | None = Field(default=None)
    latest_year: int | None = Field(default=None)


class DocumentView(BaseModel):
    """Primary document payload."""

    doc_id: str
    path: str
    raw_text: str
    payload: Any | None = None
    question: str
    word_count: int = Field(..., ge=0)


@dataclass
class DocumentSignals:
    """Aggregated document hints for prompting."""

    years: list[int]
    region_counts: Counter[str]
    theme_counts: Counter[str]
    list_counts: Counter[str]

    def to_metadata(self) -> dict[str, Any]:
        sorted_regions = sorted(
            self.region_counts.items(), key=lambda item: (-item[1], item[0])
        )
        sorted_themes = sorted(
            self.theme_counts.items(), key=lambda item: (-item[1], item[0])
        )
        sorted_lists = sorted(
            self.list_counts.items(), key=lambda item: (-item[1], item[0])
        )
        return {
            "year_min": min(self.years) if self.years else None,
            "year_max": max(self.years) if self.years else None,
            "regions": sorted_regions[:8],
            "themes": sorted_themes[:10],
            "list_hints": sorted_lists[:8],
        }


def run_iteration_temporal_synth(
    task_override: str | None = None,
) -> AgentRunSummary:
    """
    Temporal synthesis benchmark that:
      1. Reads the raw document referenced by the task.
      2. Performs heuristic snippet selection to surface timeline-rich evidence.
      3. Uses Bedrock twice: first to consolidate timeline insights, then to produce
         a Markdown summary aligned with benchmark scoring.
    """
    goal = _normalize_goal(task_override or constants.task)
    file_path, question = _extract_file_and_question(goal)
    view = _load_document(file_path, question)

    snippets, signals = _collect_snippets(view)
    ranked = _select_ranked_snippets(snippets, question, top_k=20)
    if not ranked:
        raise ValueError("Temporal synth pipeline failed: no ranked snippets available.")

    client = BedrockClient(model_id=constants.MODEL_ID)

    structured = _build_structured_summary(client, view, ranked, signals)
    final_answer = _compose_markdown_summary(client, view, structured)

    usage_payload = client.get_total_usage().model_dump()
    metadata = {
        "final_answer": final_answer,
        "question": question,
        "source_file": str(file_path),
        "ranked_snippet_count": len(ranked),
        "candidate_snippet_count": len(snippets),
        "signals": signals.to_metadata(),
        "structured_keys": list(structured.keys()),
    }
    return AgentRunSummary.from_usage(usage=usage_payload, metadata=metadata)


def _normalize_goal(goal: str) -> str:
    normalized = goal.strip()
    if not normalized:
        raise ValueError("Task instruction is empty.")
    return normalized


def _extract_file_and_question(goal: str) -> tuple[Path, str]:
    lowered = goal.lower()
    marker = "file "
    if marker not in lowered:
        raise ValueError("Temporal synth requires a task that references a source file.")
    idx = lowered.index(marker) + len(marker)
    remainder = goal[idx:].strip()
    if not remainder:
        raise ValueError("Task is missing a file path.")

    path_token, *maybe_question = remainder.split()
    file_path = Path(path_token)
    question = goal[: lowered.index(marker)].strip()
    if not question:
        question = f"Summarize the key developments in {file_path.name}"
    return file_path, question


def _load_document(path: Path, question: str) -> DocumentView:
    full_path = (PROJECT_ROOT / path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Referenced document not found: {full_path}")
    raw_text = full_path.read_text(encoding="utf-8")
    payload = _try_parse_json(raw_text)
    return DocumentView(
        doc_id=full_path.name,
        path=str(path),
        raw_text=raw_text,
        payload=payload,
        question=question,
        word_count=len(raw_text.split()),
    )


def _try_parse_json(raw_text: str) -> Any | None:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def _collect_snippets(view: DocumentView) -> tuple[list[str], DocumentSignals]:
    signals = DocumentSignals(years=[], region_counts=Counter(), theme_counts=Counter(), list_counts=Counter())
    snippets: list[str] = []

    if view.payload is not None:
        for snippet, list_path in _iter_json_snippets(view.payload):
            snippets.append(snippet)
            if list_path:
                signals.list_counts[list_path] += 1
            _register_signals(snippet, signals)
    else:
        for paragraph in _split_text(view.raw_text):
            snippets.append(paragraph)
            _register_signals(paragraph, signals)

    return snippets, signals


def _iter_json_snippets(payload: Any, base_path: str = "") -> Iterator[tuple[str, str | None]]:
    """Flatten JSON into path-prefixed snippets."""
    stack: list[tuple[str, Any]] = [(base_path, payload)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{path}.{key}" if path else str(key)
                stack.append((next_path, value))
        elif isinstance(node, list):
            list_path = path or "root"
            for idx, value in enumerate(node[:200]):
                next_path = f"{path}[{idx}]" if path else f"[{idx}]"
                stack.append((next_path, value))
            yield (f"{list_path} list with {len(node)} entries.", list_path)
        else:
            scalar = _format_scalar(node)
            if scalar:
                label = path or "value"
                snippet = f"{label}: {scalar}"
                yield (snippet, None)


def _split_text(raw_text: str) -> Iterable[str]:
    for block in re.split(r"\n\s*\n", raw_text):
        candidate = block.strip()
        if candidate:
            yield candidate


def _format_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        return compact if compact else None
    return None


def _register_signals(snippet: str, signals: DocumentSignals) -> None:
    years = [int(match) for match in DATE_PATTERN.findall(snippet)]
    for year in years:
        if 500 <= year <= 2100:
            signals.years.append(year)

    lowered = snippet.lower()
    for pattern, label in REGION_KEYWORDS.items():
        if pattern in lowered:
            signals.region_counts[label] += 1
    for pattern, label in THEME_KEYWORDS.items():
        if pattern in lowered:
            signals.theme_counts[label] += 1


def _select_ranked_snippets(
    snippets: Sequence[str],
    question: str,
    top_k: int,
) -> list[RankedSlice]:
    query_terms = set(TERM_PATTERN.findall(question.lower()))
    ranked: list[RankedSlice] = []
    for snippet in snippets:
        score = _snippet_score(snippet, query_terms)
        if score <= 0.0:
            continue
        years = [int(match) for match in DATE_PATTERN.findall(snippet) if 500 <= int(match) <= 2100]
        ranked.append(
            RankedSlice(
                text=snippet,
                score=score,
                token_count=len(snippet.split()),
                earliest_year=min(years) if years else None,
                latest_year=max(years) if years else None,
            )
        )

    ranked.sort(key=lambda item: (-item.score, -item.token_count))
    return ranked[:top_k]


def _snippet_score(snippet: str, query_terms: set[str]) -> float:
    tokens = set(TERM_PATTERN.findall(snippet.lower()))
    if not tokens:
        return 0.0
    overlap = len(tokens & query_terms)
    lexical = overlap / max(1, len(query_terms))
    richness = min(len(tokens) / 120.0, 1.0)
    year_bonus = 0.35 if DATE_PATTERN.search(snippet) else 0.0
    proper_noun_bonus = 0.25 if _has_proper_noun(snippet) else 0.0
    diversity_penalty = 0.15 if len(tokens) < 6 else 0.0
    return max(0.0, lexical * (1.0 + 0.6 * richness) + year_bonus + proper_noun_bonus - diversity_penalty)


def _has_proper_noun(snippet: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", snippet))


def _build_structured_summary(
    client: BedrockClient,
    view: DocumentView,
    ranked: Sequence[RankedSlice],
    signals: DocumentSignals,
) -> dict[str, Any]:
    evidence_block = "\n".join(
        f"{idx+1}. score={slice.score:.3f} | {slice.text}"
        for idx, slice in enumerate(ranked)
    )
    signal_metadata = signals.to_metadata()
    prompt = (
        f"Goal: {view.question}\n"
        f"Document path: {view.path}\n"
        "You are synthesizing historical evidence extracted from the source document. "
        "Use only the provided snippets. Identify the chronological span, recurring regions, "
        "dominant themes, and notable events that recur.\n\n"
        f"Document stats: {json.dumps({'word_count': view.word_count, **signal_metadata})}\n\n"
        "Snippets:\n"
        f"{evidence_block}\n\n"
        "Respond in JSON with keys: "
        "'timeline' (list of ordered notes), "
        "'regions' (list of region insights), "
        "'themes' (list of theme insights), "
        "'notable_events' (list of high-salience events), "
        "'dataset_facts' (bulletable facts about volume, coverage, or structure). "
        "Do not add commentary outside JSON."
    )
    response = client.invoke_model(
        prompt=prompt,
        system_prompt="You convert primary-source snippets into precise, verifiable JSON notes.",
        max_tokens=260,
        temperature=0.0,
    )
    parsed = _parse_json_response(response.content)
    missing_keys = {
        key
        for key in ("timeline", "regions", "themes", "notable_events", "dataset_facts")
        if key not in parsed
    }
    if missing_keys:
        raise ValueError(f"Structured summary missing keys: {sorted(missing_keys)}")
    return parsed


def _compose_markdown_summary(
    client: BedrockClient,
    view: DocumentView,
    structured: dict[str, Any],
) -> str:
    prompt = (
        f"Goal: {view.question}\n"
        f"Document path: {view.path}\n"
        "Structured evidence extracted from the document:\n"
        f"{json.dumps(structured, indent=2)}\n\n"
        "Write a Markdown summary with sections: "
        "1) Dataset Overview, 2) Structure, 3) Geographic Coverage, "
        "4) Major Themes, 5) Notable Events. "
        "Base every claim on the structured evidence. Mention the chronological span and dataset scale "
        "if available. Keep the tone factual and concise."
    )
    response = client.invoke_model(
        prompt=prompt,
        system_prompt="You produce accurate, well-structured historical summaries.",
        max_tokens=340,
        temperature=0.0,
    )
    summary = response.content.strip()
    if not summary:
        raise ValueError("Temporal synth pipeline failed: final Bedrock response is empty.")
    return summary


def _parse_json_response(text: str) -> dict[str, Any]:
    block = _extract_json_block(text)
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse Bedrock JSON payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Structured response must be a JSON object.")
    return parsed


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in Bedrock response.")
    return match.group(0)


if __name__ == "__main__":
    summary = run_iteration_temporal_synth()
    print(summary.model_dump())

