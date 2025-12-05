"""Markov Blanket: Find minimal covering set of facts for maximum F1/token.

Core insight: In probability theory, the Markov Blanket of a variable is the 
minimal set of variables that renders it conditionally independent of all others.
Applied to documents: find the minimal set of facts that "shields" all key 
information - anything outside the blanket is redundant given the blanket.

Key innovations:
1. Dependency graph construction - map which facts overlap/support each other
2. Coverage computation - track which document tokens each fact "covers"
3. Greedy blanket construction - iteratively add facts that maximize new coverage
4. Redundancy elimination - remove facts whose coverage is subsumed by others
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
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

NOISE_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "they", "to", "was", "were", "with", "this", "these", "those", "will", "can",
    "could", "may", "might", "about", "there", "here", "than", "then", "also",
    "such", "very", "only", "some", "any", "been", "being", "would", "should",
})

SYSTEM_PROMPT = (
    "You are Markov Blanket, an information compressor that finds the minimal "
    "covering set of facts. Given the essential facts that cover all key information, "
    "produce a dense bullet summary. Every token must carry information. "
    "Preserve numbers and names exactly. No speculation."
)


class CoverageFact(BaseModel):
    """Fact with its coverage set - which document tokens it represents."""
    text: str
    tokens: int = Field(..., ge=1)
    coverage: frozenset[str] = Field(..., description="Set of doc tokens this fact covers")
    coverage_score: float = Field(..., ge=0.0)
    source: str


class DocumentView(BaseModel):
    """Document with token vocabulary for coverage analysis."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)
    vocabulary: frozenset[str] = Field(default=frozenset())


def run_iteration_markov_blanket(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    vocabulary = _build_vocabulary(doc.raw_text)
    doc = doc.model_copy(update={"vocabulary": vocabulary})
    
    facts = _extract_coverage_facts(doc)
    if not facts:
        raise ValueError("Markov Blanket: no facts extracted")
    
    context_budget = _context_budget(doc.word_count)
    blanket = _construct_markov_blanket(facts, vocabulary, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, blanket, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.05,
    )
    
    usage = response.usage.model_dump()
    blanket_coverage = _compute_coverage_ratio(blanket, vocabulary)
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "vocabulary_size": len(vocabulary),
        "facts_extracted": len(facts),
        "blanket_size": len(blanket),
        "blanket_coverage": round(blanket_coverage, 3),
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


def _build_vocabulary(text: str) -> frozenset[str]:
    """Build vocabulary of meaningful tokens from document."""
    tokens = TOKEN_RE.findall(text.lower())
    meaningful = [t for t in tokens if t not in NOISE_WORDS and len(t) > 2]
    return frozenset(meaningful)


def _extract_coverage_facts(doc: DocumentView) -> list[CoverageFact]:
    """Extract facts with their coverage sets."""
    if doc.payload is not None:
        return _extract_json_coverage(doc.payload, doc.vocabulary)
    return _extract_text_coverage(doc.raw_text, doc.vocabulary)


def _extract_json_coverage(
    payload: Any,
    vocab: frozenset[str],
    path: tuple[str, ...] = (),
    depth: int = 0
) -> list[CoverageFact]:
    """Extract facts from JSON with coverage computation."""
    if depth > 8:
        return []
    
    facts: list[CoverageFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:70]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                facts.extend(_extract_json_coverage(value, vocab, new_path, depth + 1))
            else:
                fact = _make_coverage_fact(new_path, value, vocab)
                if fact:
                    facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:15]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_coverage(item, vocab, new_path, depth + 1))
            else:
                fact = _make_coverage_fact(new_path, item, vocab)
                if fact:
                    facts.append(fact)
    
    return facts


def _make_coverage_fact(path: tuple[str, ...], value: Any, vocab: frozenset[str]) -> CoverageFact | None:
    """Create fact with computed coverage set."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    
    formatted = _format_value(value)
    if not formatted or len(formatted) < 2:
        return None
    
    label = ".".join(path) if path else "value"
    text = f"{label}: {formatted}"
    
    text_tokens = set(TOKEN_RE.findall(text.lower())) - NOISE_WORDS
    coverage = vocab & text_tokens
    
    if not coverage:
        return None
    
    tokens = len(text.split())
    if tokens < 2 or tokens > 45:
        return None
    
    coverage_density = len(coverage) / max(1, tokens)
    numeric_bonus = 1.5 if any(c.isdigit() for c in formatted) else 0.4
    depth_penalty = 1.0 + len(path) * 0.12
    
    coverage_score = (len(coverage) * 0.5 + coverage_density * 3.0 + numeric_bonus) / depth_penalty
    
    return CoverageFact(
        text=text,
        tokens=tokens,
        coverage=frozenset(coverage),
        coverage_score=coverage_score,
        source="json",
    )


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().split())[:180]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)[:180] if value else ""


def _extract_text_coverage(text: str, vocab: frozenset[str]) -> list[CoverageFact]:
    """Extract sentence facts with coverage sets."""
    sentences = _split_sentences(text)
    facts: list[CoverageFact] = []
    
    for idx, sentence in enumerate(sentences):
        clean = " ".join(sentence.split())
        tokens = len(clean.split())
        
        if tokens < 5 or tokens > 55:
            continue
        
        sentence_tokens = set(TOKEN_RE.findall(clean.lower())) - NOISE_WORDS
        coverage = vocab & sentence_tokens
        
        if not coverage:
            continue
        
        coverage_density = len(coverage) / max(1, tokens)
        numeric_bonus = 1.0 if any(c.isdigit() for c in clean) else 0.2
        position_factor = 1.0 / (1.0 + idx * 0.04)
        
        coverage_score = (len(coverage) * 0.6 + coverage_density * 2.5 + numeric_bonus) * position_factor
        
        facts.append(CoverageFact(
            text=clean,
            tokens=tokens,
            coverage=frozenset(coverage),
            coverage_score=coverage_score,
            source="text",
        ))
    
    return facts


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [s.strip() for s in parts if s.strip()]


def _construct_markov_blanket(
    facts: list[CoverageFact],
    vocabulary: frozenset[str],
    budget: int,
) -> list[CoverageFact]:
    """Construct minimal covering set (Markov Blanket) via greedy set cover."""
    blanket: list[CoverageFact] = []
    covered: set[str] = set()
    tokens_used = 0
    seen_text: set[str] = set()
    
    remaining = list(facts)
    
    while remaining and tokens_used < budget and len(blanket) < 30:
        best_fact = None
        best_marginal = -1.0
        
        for fact in remaining:
            if fact.text.lower() in seen_text:
                continue
            if tokens_used + fact.tokens > budget:
                continue
            
            new_coverage = fact.coverage - covered
            marginal_value = len(new_coverage) / max(1, fact.tokens)
            
            if marginal_value > best_marginal:
                best_marginal = marginal_value
                best_fact = fact
        
        if best_fact is None or best_marginal <= 0:
            break
        
        blanket.append(best_fact)
        covered.update(best_fact.coverage)
        seen_text.add(best_fact.text.lower())
        tokens_used += best_fact.tokens
        remaining.remove(best_fact)
    
    return blanket


def _compute_coverage_ratio(blanket: list[CoverageFact], vocab: frozenset[str]) -> float:
    """Compute what fraction of vocabulary the blanket covers."""
    if not vocab:
        return 1.0
    covered = set()
    for fact in blanket:
        covered.update(fact.coverage)
    return len(covered) / len(vocab)


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 200
    base = math.log(word_count + 20, 1.65) * 90
    return int(max(180, min(420, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 50
    base = math.sqrt(word_count) * 1.7
    return int(max(46, min(78, base)))


def _build_prompt(doc: DocumentView, blanket: list[CoverageFact], output_budget: int) -> str:
    """Build prompt from Markov Blanket facts."""
    lines = [
        f"Task: {doc.goal}",
        f"Source: {Path(doc.path).name}",
        "",
        f"Produce a {output_budget}-token bullet summary from this minimal covering set:",
        "",
    ]
    
    for fact in blanket:
        lines.append(f"• {fact.text}")
    
    lines.extend([
        "",
        "Output bullets only. Preserve all numbers. No extra commentary.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_markov_blanket()
    print(summary.model_dump())

