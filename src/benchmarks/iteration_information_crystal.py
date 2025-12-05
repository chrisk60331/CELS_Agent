"""Information Crystal: Crystallize facts by information-theoretic density.

Core insight: Information theory tells us that the "surprise" value of a message 
is proportional to how unexpected it is. By computing the information content 
(self-information) of each fact and selecting facts with maximum information 
density per token, we achieve optimal F1/token compression.

Key innovations:
1. Self-information scoring - -log(p) for each term based on corpus frequency
2. Information density - total self-information / token count
3. Crystal formation - greedily add facts maximizing marginal information gain
4. Redundancy pruning - skip facts whose information is subsumed by selected facts
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

FILLER = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "they", "to", "was", "were", "with", "this", "these", "those", "will", "can",
    "could", "may", "might", "about", "there", "here", "than", "then", "also",
    "such", "very", "only", "some", "any", "been", "being", "would", "should",
})

CONTEXT_CEILING = 340
OUTPUT_CEILING = 66

SYSTEM_PROMPT = (
    "You are Information Crystal, an entropy-optimizing compressor. Given facts "
    "selected for maximum information density, synthesize the densest bullet "
    "summary possible. Every word must maximize information content. "
    "Preserve numbers and names exactly. Zero speculation."
)


class CrystalFact(BaseModel):
    """Fact with information-theoretic metrics."""
    text: str
    tokens: int = Field(..., ge=1)
    info_content: float = Field(..., ge=0.0, description="Total self-information")
    info_density: float = Field(..., ge=0.0, description="Info per token")
    unique_terms: frozenset[str] = Field(...)
    source: str


class DocumentView(BaseModel):
    """Document with corpus statistics."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)
    term_probs: dict[str, float] = Field(default_factory=dict)


def run_iteration_information_crystal(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    term_probs = _compute_term_probabilities(doc.raw_text)
    doc = doc.model_copy(update={"term_probs": term_probs})
    
    facts = _extract_crystal_facts(doc)
    
    if not facts:
        raise ValueError("Information Crystal: no facts extracted")
    
    context_budget = _context_budget(doc.word_count)
    crystal = _form_crystal(facts, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, crystal, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.04,
    )
    
    usage = response.usage.model_dump()
    total_info = sum(f.info_content for f in crystal)
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "facts_extracted": len(facts),
        "crystal_size": len(crystal),
        "total_information": round(total_info, 2),
        "avg_info_density": round(total_info / max(1, sum(f.tokens for f in crystal)), 3),
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


def _compute_term_probabilities(text: str) -> dict[str, float]:
    """Compute empirical probability of each term in the corpus."""
    tokens = [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in FILLER and len(t) > 2]
    
    if not tokens:
        return {}
    
    counts = Counter(tokens)
    total = sum(counts.values())
    
    return {term: count / total for term, count in counts.items()}


def _self_information(prob: float) -> float:
    """Compute self-information: -log2(p). Higher = more surprising."""
    if prob <= 0:
        return 10.0
    return -math.log2(prob)


def _extract_crystal_facts(doc: DocumentView) -> list[CrystalFact]:
    """Extract facts with information-theoretic scoring."""
    if doc.payload is not None:
        return _extract_json_crystals(doc.payload, doc.term_probs)
    return _extract_text_crystals(doc.raw_text, doc.term_probs)


def _extract_json_crystals(payload: Any, term_probs: dict[str, float], path: tuple[str, ...] = (), depth: int = 0) -> list[CrystalFact]:
    """Extract facts from JSON with information scoring."""
    if depth > 8:
        return []
    
    facts: list[CrystalFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:65]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                facts.extend(_extract_json_crystals(value, term_probs, new_path, depth + 1))
            else:
                fact = _make_crystal_fact(new_path, value, term_probs, "json")
                if fact:
                    facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:14]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_crystals(item, term_probs, new_path, depth + 1))
            else:
                fact = _make_crystal_fact(new_path, item, term_probs, "json")
                if fact:
                    facts.append(fact)
    
    return facts


def _make_crystal_fact(path: tuple[str, ...], value: Any, term_probs: dict[str, float], source: str) -> CrystalFact | None:
    """Create fact with information-theoretic metrics."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    
    formatted = _format_value(value)
    if not formatted:
        return None
    
    label = ".".join(path) if path else "value"
    text = f"{label}: {formatted}"
    
    terms = {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in FILLER and len(t) > 2}
    
    if not terms:
        return None
    
    tokens = len(text.split())
    if tokens < 2 or tokens > 42:
        return None
    
    info_content = sum(_self_information(term_probs.get(t, 0.001)) for t in terms)
    
    numeric_bonus = 2.0 if any(c.isdigit() for c in formatted) else 0.5
    info_content += numeric_bonus
    
    info_density = info_content / tokens
    
    return CrystalFact(
        text=text,
        tokens=tokens,
        info_content=info_content,
        info_density=info_density,
        unique_terms=frozenset(terms),
        source=source,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().split())[:170]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)[:170] if value else ""


def _extract_text_crystals(text: str, term_probs: dict[str, float]) -> list[CrystalFact]:
    """Extract sentence facts with information scoring."""
    sentences = _split_sentences(text)
    facts: list[CrystalFact] = []
    
    for idx, sentence in enumerate(sentences):
        clean = " ".join(sentence.split())
        
        terms = {t.lower() for t in TOKEN_RE.findall(clean) if t.lower() not in FILLER and len(t) > 2}
        
        if not terms:
            continue
        
        tokens = len(clean.split())
        if tokens < 5 or tokens > 50:
            continue
        
        info_content = sum(_self_information(term_probs.get(t, 0.001)) for t in terms)
        
        numeric_bonus = 1.5 if any(c.isdigit() for c in clean) else 0.3
        position_factor = 1.0 / (1.0 + idx * 0.02)
        
        info_content = (info_content + numeric_bonus) * position_factor
        info_density = info_content / tokens
        
        facts.append(CrystalFact(
            text=clean,
            tokens=tokens,
            info_content=info_content,
            info_density=info_density,
            unique_terms=frozenset(terms),
            source="text",
        ))
    
    return facts


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [s.strip() for s in parts if s.strip()]


def _form_crystal(facts: list[CrystalFact], budget: int) -> list[CrystalFact]:
    """Form optimal information crystal via greedy selection with redundancy pruning."""
    crystal: list[CrystalFact] = []
    tokens_used = 0
    covered_terms: set[str] = set()
    seen_text: set[str] = set()
    
    remaining = list(facts)
    
    while remaining and tokens_used < budget and len(crystal) < 28:
        best_fact = None
        best_marginal = -1.0
        
        for fact in remaining:
            if fact.text.lower() in seen_text:
                continue
            if tokens_used + fact.tokens > budget:
                continue
            
            new_terms = fact.unique_terms - covered_terms
            marginal_info = sum(_self_information(0.05) for _ in new_terms)
            marginal_density = marginal_info / max(1, fact.tokens)
            
            if marginal_density > best_marginal:
                best_marginal = marginal_density
                best_fact = fact
        
        if best_fact is None or best_marginal <= 0.1:
            break
        
        crystal.append(best_fact)
        covered_terms.update(best_fact.unique_terms)
        seen_text.add(best_fact.text.lower())
        tokens_used += best_fact.tokens
        remaining.remove(best_fact)
    
    return crystal


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 180
    base = math.log(word_count + 18, 1.7) * 82
    return int(max(170, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 44
    base = math.sqrt(word_count) * 1.55
    return int(max(40, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, crystal: list[CrystalFact], output_budget: int) -> str:
    """Build prompt from information crystal."""
    lines = [
        f"Task: {doc.goal}",
        f"Source: {Path(doc.path).name}",
        "",
        f"High-information facts selected. Synthesize {output_budget}-token bullet summary:",
        "",
    ]
    
    for fact in crystal:
        lines.append(f"⬢ {fact.text}")
    
    lines.extend([
        "",
        "Dense bullets only. Numbers exact. Zero filler.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_information_crystal()
    print(summary.model_dump())

