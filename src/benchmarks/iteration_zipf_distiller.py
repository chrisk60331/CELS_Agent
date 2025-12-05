"""Zipf Distiller: Leverage Zipfian distribution for maximum F1/token compression.

Core insight: In any document, information follows Zipf's law. The "head" of the
distribution contains high-frequency meaningful concepts; the "long tail" has 
low-value noise. By extracting the Zipfian head and synthesizing a minimal
representation, we achieve maximal information density.

Key innovations:
1. Zipf rank scoring - weight terms by their inverse rank position
2. Head-tail separation - identify the elbow in the frequency distribution  
3. Concept coalescence - merge related high-rank terms into unified facts
4. Minimal prompt synthesis - generate only from the essential concept core
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "they", "to", "was", "were", "with", "this", "these", "those", "will", "can",
    "could", "may", "might", "about", "there", "here", "than", "then", "after",
    "before", "between", "while", "where", "when", "which", "been", "being",
    "would", "should", "shall", "also", "any", "some", "such", "very", "only",
})

ZIPF_HEAD_RATIO = 0.15
MIN_FACTS = 8
MAX_FACTS = 28
CONTEXT_TOKENS_CEILING = 380
OUTPUT_TOKENS_CEILING = 72

SYSTEM_PROMPT = (
    "You are Zipf Distiller, an information compression engine that maximizes "
    "F1 score per token. Given high-value concept facts extracted from a document, "
    "produce a dense bullet summary. Include all numbers, names, and key relationships. "
    "Never speculate. Use only provided evidence."
)


class ZipfTerm(BaseModel):
    """Term with Zipfian rank and frequency metadata."""
    term: str = Field(..., description="Normalized term")
    frequency: int = Field(..., ge=1)
    rank: int = Field(..., ge=1)
    zipf_score: float = Field(..., ge=0.0, description="Information value score")


class ConceptFact(BaseModel):
    """Atomic fact extracted from document."""
    text: str = Field(..., description="Fact statement")
    tokens: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0)
    source: str = Field(..., description="Extraction source type")


class DocumentView(BaseModel):
    """Normalized document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)
    zipf_terms: tuple[ZipfTerm, ...] = Field(default=())


def run_iteration_zipf_distiller(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    zipf_terms = _compute_zipf_distribution(doc.raw_text)
    doc = doc.model_copy(update={"zipf_terms": zipf_terms})
    
    head_terms = _extract_zipf_head(zipf_terms)
    facts = _extract_facts(doc, head_terms)
    
    if not facts:
        raise ValueError("Zipf Distiller: no facts extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_facts(facts, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, selected, head_terms, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.08,
    )
    
    usage = response.usage.model_dump()
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "zipf_head_size": len(head_terms),
        "facts_extracted": len(facts),
        "facts_selected": len(selected),
        "context_budget": context_budget,
        "output_budget": output_budget,
    }
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


def _require_goal(goal: str) -> str:
    normalized = goal.strip()
    if not normalized:
        raise ValueError("Benchmark task required")
    return normalized


def _load_document(goal: str) -> DocumentView:
    file_path = _extract_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Source file missing: {full_path}")
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
        raise ValueError("No file path in goal")
    return Path(remainder.split()[0])


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _compute_zipf_distribution(text: str) -> tuple[ZipfTerm, ...]:
    """Compute Zipfian ranking of all terms in the document."""
    tokens = [
        t.lower() for t in TOKEN_PATTERN.findall(text)
        if t.lower() not in STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return ()
    
    counts = Counter(tokens)
    ranked = counts.most_common()
    
    terms = []
    for rank, (term, freq) in enumerate(ranked, start=1):
        zipf_score = freq / math.log(rank + 1)
        terms.append(ZipfTerm(
            term=term,
            frequency=freq,
            rank=rank,
            zipf_score=zipf_score,
        ))
    return tuple(terms)


def _extract_zipf_head(terms: Sequence[ZipfTerm]) -> list[str]:
    """Extract terms in the Zipfian head (high-value information core)."""
    if not terms:
        return []
    
    head_count = max(5, int(len(terms) * ZIPF_HEAD_RATIO))
    head_count = min(head_count, 30)
    
    head_terms = [t.term for t in sorted(terms, key=lambda x: -x.zipf_score)[:head_count]]
    return head_terms


def _extract_facts(doc: DocumentView, head_terms: list[str]) -> list[ConceptFact]:
    """Extract atomic facts prioritizing Zipfian head terms."""
    facts: list[ConceptFact] = []
    head_set = set(t.lower() for t in head_terms)
    
    if doc.payload is not None:
        facts.extend(_extract_json_facts(doc.payload, head_set))
    else:
        facts.extend(_extract_text_facts(doc.raw_text, head_set))
    
    return facts


def _extract_json_facts(payload: Any, head_terms: set[str], path: tuple[str, ...] = (), depth: int = 0) -> list[ConceptFact]:
    """Recursively extract facts from JSON, scoring by head term overlap."""
    if depth > 8 or len(path) > 6:
        return []
    
    facts: list[ConceptFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:60]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                facts.extend(_extract_json_facts(value, head_terms, new_path, depth + 1))
            else:
                fact = _make_scalar_fact(new_path, value, head_terms)
                if fact:
                    facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:12]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_facts(item, head_terms, new_path, depth + 1))
            else:
                fact = _make_scalar_fact(new_path, item, head_terms)
                if fact:
                    facts.append(fact)
    
    return facts


def _make_scalar_fact(path: tuple[str, ...], value: Any, head_terms: set[str]) -> ConceptFact | None:
    """Create a fact from a scalar value, scoring by head term overlap."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    
    formatted = _format_value(value)
    if not formatted or len(formatted) < 2:
        return None
    
    label = ".".join(path) if path else "value"
    text = f"{label}: {formatted}"
    
    text_lower = text.lower()
    overlap = sum(1 for t in head_terms if t in text_lower)
    
    numeric_bonus = 1.2 if any(c.isdigit() for c in formatted) else 0.3
    length_factor = min(1.0, len(formatted) / 50)
    depth_penalty = 1.0 + len(path) * 0.15
    
    score = (2.5 + overlap * 0.8 + numeric_bonus + length_factor) / depth_penalty
    
    tokens = len(text.split())
    if tokens < 2 or tokens > 50:
        return None
    
    return ConceptFact(text=text, tokens=tokens, score=score, source="json")


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().split())[:200]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)[:200] if value else ""


def _extract_text_facts(text: str, head_terms: set[str]) -> list[ConceptFact]:
    """Extract facts from plain text by sentence analysis."""
    sentences = _split_sentences(text)
    facts: list[ConceptFact] = []
    
    for idx, sentence in enumerate(sentences):
        clean = " ".join(sentence.split())
        tokens = len(clean.split())
        
        if tokens < 5 or tokens > 60:
            continue
        
        sentence_lower = clean.lower()
        overlap = sum(1 for t in head_terms if t in sentence_lower)
        
        numeric_bonus = 0.9 if any(c.isdigit() for c in clean) else 0.2
        position_factor = 1.0 / (1.0 + idx * 0.05)
        
        score = (3.0 + overlap * 0.9 + numeric_bonus) * position_factor
        
        facts.append(ConceptFact(
            text=clean,
            tokens=tokens,
            score=score,
            source="text",
        ))
    
    return facts


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [s.strip() for s in sentences if s.strip()]


def _select_facts(facts: list[ConceptFact], budget: int) -> list[ConceptFact]:
    """Greedy selection maximizing score/token ratio within budget."""
    ranked = sorted(facts, key=lambda f: (-f.score / max(1, f.tokens), -f.score))
    
    selected: list[ConceptFact] = []
    tokens_used = 0
    seen: set[str] = set()
    
    for fact in ranked:
        if len(selected) >= MAX_FACTS:
            break
        
        normalized = fact.text.lower()
        if normalized in seen:
            continue
        
        if tokens_used + fact.tokens > budget:
            continue
        
        selected.append(fact)
        seen.add(normalized)
        tokens_used += fact.tokens
    
    if len(selected) < MIN_FACTS:
        for fact in ranked:
            if len(selected) >= MIN_FACTS:
                break
            normalized = fact.text.lower()
            if normalized not in seen:
                selected.append(fact)
                seen.add(normalized)
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 180
    base = math.log(word_count + 15, 1.6) * 85
    return int(max(160, min(CONTEXT_TOKENS_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 48
    base = math.sqrt(word_count) * 1.6
    return int(max(44, min(OUTPUT_TOKENS_CEILING, base)))


def _build_prompt(doc: DocumentView, facts: list[ConceptFact], head_terms: list[str], output_budget: int) -> str:
    """Build minimal prompt emphasizing high-value facts."""
    lines = [
        f"Task: {doc.goal}",
        f"Source: {Path(doc.path).name}",
        f"Core concepts: {', '.join(head_terms[:10])}",
        "",
        f"Generate a {output_budget}-token maximum bullet summary from these facts:",
        "",
    ]
    
    for fact in facts:
        lines.append(f"• {fact.text}")
    
    lines.extend([
        "",
        "Rules: bullets only, preserve numbers exactly, no speculation.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_zipf_distiller()
    print(summary.model_dump())

