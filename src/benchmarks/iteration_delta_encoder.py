"""Delta Encoder: Extract only novel information via incremental processing.

Core insight: Documents contain massive redundancy. As we traverse the document,
only a fraction of each new segment adds novel information. By tracking what's
already "covered" and extracting only deltas (new information), we achieve 
maximum compression with minimal loss.

Key innovations:
1. Running coverage set - track all concepts seen so far
2. Delta extraction - for each segment, extract only novel information
3. Novelty scoring - prioritize high-novelty deltas
4. Compression synthesis - generate summary from deltas only
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

NOISE = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "they", "to", "was", "were", "with", "this", "these", "those", "will", "can",
    "could", "may", "might", "about", "there", "here", "than", "then", "also",
    "such", "very", "only", "some", "any", "been", "being", "would", "should",
})

MIN_NOVELTY_RATIO = 0.15
CONTEXT_CEILING = 350
OUTPUT_CEILING = 68

SYSTEM_PROMPT = (
    "You are Delta Encoder, an information compressor that captures only novel "
    "information deltas. Given extracted deltas representing unique information, "
    "synthesize the densest possible bullet summary. Every token must be essential. "
    "Preserve numbers and names exactly. No speculation."
)


class DeltaFact(BaseModel):
    """Fact representing novel information delta."""
    text: str
    tokens: int = Field(..., ge=1)
    novel_terms: frozenset[str] = Field(..., description="Terms that were novel when extracted")
    novelty_ratio: float = Field(..., ge=0.0, le=1.0)
    delta_score: float = Field(..., ge=0.0)
    source: str


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_delta_encoder(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    deltas = _extract_deltas(doc)
    
    if not deltas:
        raise ValueError("Delta Encoder: no deltas extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_deltas(deltas, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, selected, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.05,
    )
    
    usage = response.usage.model_dump()
    total_novel = sum(len(d.novel_terms) for d in selected)
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "deltas_extracted": len(deltas),
        "deltas_selected": len(selected),
        "total_novel_terms": total_novel,
        "context_budget": context_budget,
        "output_budget": output_budget,
    }
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


def _require_goal(goal: str) -> str:
    normalized = goal.strip()
    if not normalized:
        raise ValueError("Task required")
    return normalized


def _load_document(goal: str) -> DocumentView:
    file_path = _extract_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Missing: {full_path}")
    raw_text = full_path.read_text(encoding="utf-8")
    payload = _try_parse_json(raw_text)
    return DocumentView(
        goal=goal,
        path=str(file_path),
        raw_text=raw_text,
        payload=payload,
        word_count=len(raw_text.split()),
    )


def _extract_file_path(goal: str) -> Path:
    marker = "file "
    lowered = goal.lower()
    if marker not in lowered:
        raise ValueError("Goal must include 'file <path>'")
    idx = lowered.index(marker) + len(marker)
    remainder = goal[idx:].strip()
    if not remainder:
        raise ValueError("No file path")
    return Path(remainder.split()[0])


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_deltas(doc: DocumentView) -> list[DeltaFact]:
    """Extract facts with novelty tracking."""
    if doc.payload is not None:
        return _extract_json_deltas(doc.payload)
    return _extract_text_deltas(doc.raw_text)


def _extract_json_deltas(payload: Any) -> list[DeltaFact]:
    """Extract deltas from JSON with running coverage."""
    seen_terms: set[str] = set()
    deltas: list[DeltaFact] = []
    
    facts = _flatten_json(payload)
    
    for path, value in facts:
        formatted = _format_value(value)
        if not formatted:
            continue
        
        label = ".".join(path) if path else "value"
        text = f"{label}: {formatted}"
        
        text_terms = {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in NOISE and len(t) > 2}
        novel_terms = text_terms - seen_terms
        
        tokens = len(text.split())
        if tokens < 2 or tokens > 45:
            continue
        
        novelty_ratio = len(novel_terms) / max(1, len(text_terms)) if text_terms else 0.0
        
        if novelty_ratio < MIN_NOVELTY_RATIO and len(novel_terms) < 2:
            seen_terms.update(text_terms)
            continue
        
        numeric_bonus = 1.4 if any(c.isdigit() for c in formatted) else 0.3
        depth_penalty = 1.0 + len(path) * 0.1
        
        delta_score = (len(novel_terms) * 0.8 + novelty_ratio * 3.0 + numeric_bonus) / depth_penalty
        
        deltas.append(DeltaFact(
            text=text,
            tokens=tokens,
            novel_terms=frozenset(novel_terms),
            novelty_ratio=novelty_ratio,
            delta_score=delta_score,
            source="json",
        ))
        
        seen_terms.update(text_terms)
    
    return deltas


def _flatten_json(payload: Any, path: tuple[str, ...] = (), depth: int = 0) -> list[tuple[tuple[str, ...], Any]]:
    """Flatten JSON to list of (path, scalar_value) pairs."""
    if depth > 8:
        return []
    
    results: list[tuple[tuple[str, ...], Any]] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:70]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                results.extend(_flatten_json(value, new_path, depth + 1))
            else:
                results.append((new_path, value))
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:15]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                results.extend(_flatten_json(item, new_path, depth + 1))
            else:
                results.append((new_path, item))
    
    else:
        results.append((path, payload))
    
    return results


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = " ".join(value.strip().split())
        return stripped[:180] if stripped else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)[:180]


def _extract_text_deltas(text: str) -> list[DeltaFact]:
    """Extract deltas from text with running coverage."""
    seen_terms: set[str] = set()
    deltas: list[DeltaFact] = []
    sentences = _split_sentences(text)
    
    for idx, sentence in enumerate(sentences):
        clean = " ".join(sentence.split())
        
        sentence_terms = {t.lower() for t in TOKEN_RE.findall(clean) if t.lower() not in NOISE and len(t) > 2}
        novel_terms = sentence_terms - seen_terms
        
        tokens = len(clean.split())
        if tokens < 5 or tokens > 55:
            seen_terms.update(sentence_terms)
            continue
        
        novelty_ratio = len(novel_terms) / max(1, len(sentence_terms)) if sentence_terms else 0.0
        
        if novelty_ratio < MIN_NOVELTY_RATIO and len(novel_terms) < 2:
            seen_terms.update(sentence_terms)
            continue
        
        numeric_bonus = 1.0 if any(c.isdigit() for c in clean) else 0.2
        position_factor = 1.0 / (1.0 + idx * 0.03)
        
        delta_score = (len(novel_terms) * 0.9 + novelty_ratio * 2.5 + numeric_bonus) * position_factor
        
        deltas.append(DeltaFact(
            text=clean,
            tokens=tokens,
            novel_terms=frozenset(novel_terms),
            novelty_ratio=novelty_ratio,
            delta_score=delta_score,
            source="text",
        ))
        
        seen_terms.update(sentence_terms)
    
    return deltas


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [s.strip() for s in parts if s.strip()]


def _select_deltas(deltas: list[DeltaFact], budget: int) -> list[DeltaFact]:
    """Select deltas by score/token efficiency."""
    ranked = sorted(deltas, key=lambda d: (-d.delta_score / max(1, d.tokens), -d.delta_score))
    
    selected: list[DeltaFact] = []
    tokens_used = 0
    seen: set[str] = set()
    
    for delta in ranked:
        if delta.text.lower() in seen:
            continue
        if tokens_used + delta.tokens > budget:
            continue
        if len(selected) >= 30:
            break
        
        selected.append(delta)
        seen.add(delta.text.lower())
        tokens_used += delta.tokens
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 180
    base = math.log(word_count + 15, 1.7) * 85
    return int(max(170, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 46
    base = math.sqrt(word_count) * 1.6
    return int(max(42, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, deltas: list[DeltaFact], output_budget: int) -> str:
    """Build prompt from deltas."""
    lines = [
        f"Task: {doc.goal}",
        f"Source: {Path(doc.path).name}",
        "",
        f"These are unique information deltas. Synthesize {output_budget}-token bullet summary:",
        "",
    ]
    
    for delta in deltas:
        lines.append(f"Δ {delta.text}")
    
    lines.extend([
        "",
        "Output dense bullets. Keep numbers exact. No speculation.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_delta_encoder()
    print(summary.model_dump())

