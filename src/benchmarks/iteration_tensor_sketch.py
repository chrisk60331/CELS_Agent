"""Tensor Sketch: Dimensionality reduction for maximum information retention.

Core insight: A document can be viewed as a high-dimensional space where each 
fact is a vector. By projecting facts into a lower-dimensional "sketch" space
and selecting representatives that span this space maximally, we achieve
optimal coverage with minimal redundancy.

Key innovations:
1. Fact vectorization - represent facts as term-frequency vectors
2. Orthogonality scoring - prefer facts that are orthogonal to already selected
3. Span maximization - greedily add facts that expand the information "span"
4. Sketch synthesis - generate from maximally spanning fact set
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
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

CONTEXT_CEILING = 355
OUTPUT_CEILING = 65

SYSTEM_PROMPT = (
    "You are Tensor Sketch, a dimensionality-reduction compressor. Given facts "
    "that maximally span the document's information space, synthesize the densest "
    "bullet summary. Every token must carry unique information. "
    "Preserve numbers and names exactly. No speculation."
)


class SketchFact(BaseModel):
    """Fact with vector representation for orthogonality computation."""
    text: str
    tokens: int = Field(..., ge=1)
    term_vector: dict[str, float] = Field(..., description="Normalized TF vector")
    magnitude: float = Field(..., ge=0.0)
    orthogonality_score: float = Field(default=0.0)
    source: str


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)
    idf_weights: dict[str, float] = Field(default_factory=dict)


def run_iteration_tensor_sketch(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    facts = _extract_sketch_facts(doc)
    
    if not facts:
        raise ValueError("Tensor Sketch: no facts extracted")
    
    idf = _compute_idf(facts)
    doc = doc.model_copy(update={"idf_weights": idf})
    
    facts = _apply_tfidf_weighting(facts, idf)
    
    context_budget = _context_budget(doc.word_count)
    sketch = _build_sketch(facts, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, sketch, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.05,
    )
    
    usage = response.usage.model_dump()
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "facts_extracted": len(facts),
        "sketch_size": len(sketch),
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


def _extract_sketch_facts(doc: DocumentView) -> list[SketchFact]:
    """Extract facts with TF vectors."""
    if doc.payload is not None:
        return _extract_json_sketches(doc.payload)
    return _extract_text_sketches(doc.raw_text)


def _extract_json_sketches(payload: Any, path: tuple[str, ...] = (), depth: int = 0) -> list[SketchFact]:
    """Extract facts from JSON with vector representations."""
    if depth > 8:
        return []
    
    facts: list[SketchFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:70]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                facts.extend(_extract_json_sketches(value, new_path, depth + 1))
            else:
                fact = _make_sketch_fact(new_path, value, "json")
                if fact:
                    facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:15]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_sketches(item, new_path, depth + 1))
            else:
                fact = _make_sketch_fact(new_path, item, "json")
                if fact:
                    facts.append(fact)
    
    return facts


def _make_sketch_fact(path: tuple[str, ...], value: Any, source: str) -> SketchFact | None:
    """Create fact with TF vector."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    
    formatted = _format_value(value)
    if not formatted:
        return None
    
    label = ".".join(path) if path else "value"
    text = f"{label}: {formatted}"
    
    terms = [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in NOISE and len(t) > 2]
    
    if not terms:
        return None
    
    tokens = len(text.split())
    if tokens < 2 or tokens > 45:
        return None
    
    term_counts = Counter(terms)
    total = sum(term_counts.values())
    term_vector = {term: count / total for term, count in term_counts.items()}
    
    magnitude = math.sqrt(sum(v * v for v in term_vector.values()))
    
    return SketchFact(
        text=text,
        tokens=tokens,
        term_vector=term_vector,
        magnitude=magnitude,
        source=source,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().split())[:175]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)[:175] if value else ""


def _extract_text_sketches(text: str) -> list[SketchFact]:
    """Extract sentence facts with vectors."""
    sentences = _split_sentences(text)
    facts: list[SketchFact] = []
    
    for sentence in sentences:
        clean = " ".join(sentence.split())
        
        terms = [t.lower() for t in TOKEN_RE.findall(clean) if t.lower() not in NOISE and len(t) > 2]
        
        if not terms:
            continue
        
        tokens = len(clean.split())
        if tokens < 5 or tokens > 55:
            continue
        
        term_counts = Counter(terms)
        total = sum(term_counts.values())
        term_vector = {term: count / total for term, count in term_counts.items()}
        
        magnitude = math.sqrt(sum(v * v for v in term_vector.values()))
        
        facts.append(SketchFact(
            text=clean,
            tokens=tokens,
            term_vector=term_vector,
            magnitude=magnitude,
            source="text",
        ))
    
    return facts


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [s.strip() for s in parts if s.strip()]


def _compute_idf(facts: list[SketchFact]) -> dict[str, float]:
    """Compute IDF weights for terms across all facts."""
    doc_freq: Counter[str] = Counter()
    
    for fact in facts:
        for term in fact.term_vector:
            doc_freq[term] += 1
    
    n_docs = len(facts)
    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(n_docs / (1 + df)) + 1
    
    return idf


def _apply_tfidf_weighting(facts: list[SketchFact], idf: dict[str, float]) -> list[SketchFact]:
    """Apply TF-IDF weighting to fact vectors."""
    weighted = []
    for fact in facts:
        new_vector = {}
        for term, tf in fact.term_vector.items():
            new_vector[term] = tf * idf.get(term, 1.0)
        
        magnitude = math.sqrt(sum(v * v for v in new_vector.values())) if new_vector else 0.0
        
        weighted.append(fact.model_copy(update={
            "term_vector": new_vector,
            "magnitude": magnitude,
        }))
    
    return weighted


def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float], mag1: float, mag2: float) -> float:
    """Compute cosine similarity between two vectors."""
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
    return dot / (mag1 * mag2)


def _build_sketch(facts: list[SketchFact], budget: int) -> list[SketchFact]:
    """Build sketch by selecting maximally orthogonal facts."""
    sketch: list[SketchFact] = []
    tokens_used = 0
    seen_text: set[str] = set()
    
    facts_by_magnitude = sorted(facts, key=lambda f: -f.magnitude)
    
    if facts_by_magnitude:
        first = facts_by_magnitude[0]
        if first.tokens <= budget:
            sketch.append(first)
            seen_text.add(first.text.lower())
            tokens_used += first.tokens
    
    remaining = [f for f in facts if f.text.lower() not in seen_text]
    
    while remaining and tokens_used < budget and len(sketch) < 30:
        best_fact = None
        best_orthogonality = -1.0
        
        for fact in remaining:
            if fact.text.lower() in seen_text:
                continue
            if tokens_used + fact.tokens > budget:
                continue
            
            max_sim = 0.0
            for selected in sketch:
                sim = _cosine_similarity(
                    fact.term_vector, selected.term_vector,
                    fact.magnitude, selected.magnitude
                )
                if sim > max_sim:
                    max_sim = sim
            
            orthogonality = (1.0 - max_sim) * fact.magnitude
            
            numeric_bonus = 1.3 if any(c.isdigit() for c in fact.text) else 0.0
            orthogonality += numeric_bonus
            
            if orthogonality > best_orthogonality:
                best_orthogonality = orthogonality
                best_fact = fact
        
        if best_fact is None or best_orthogonality <= 0.05:
            break
        
        sketch.append(best_fact)
        seen_text.add(best_fact.text.lower())
        tokens_used += best_fact.tokens
        remaining = [f for f in remaining if f.text.lower() not in seen_text]
    
    return sketch


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 185
    base = math.log(word_count + 16, 1.68) * 86
    return int(max(175, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 44
    base = math.sqrt(word_count) * 1.55
    return int(max(42, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, sketch: list[SketchFact], output_budget: int) -> str:
    """Build prompt from tensor sketch."""
    lines = [
        f"Task: {doc.goal}",
        f"Source: {Path(doc.path).name}",
        "",
        f"Facts spanning document information space. Generate {output_budget}-token summary:",
        "",
    ]
    
    for fact in sketch:
        lines.append(f"▸ {fact.text}")
    
    lines.extend([
        "",
        "Dense bullets only. Exact numbers. No filler.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_tensor_sketch()
    print(summary.model_dump())

