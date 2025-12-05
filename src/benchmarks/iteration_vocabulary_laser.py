"""Vocabulary Laser: Precision extraction for maximum F1/token ratio.

Core insight: F1 score measures vocabulary overlap. Instead of trying to summarize
everything, laser-focus on extracting the exact tokens that matter: brand names,
product names, numbers with units, percentages, grades, and key categorical terms.

Key innovations:
1. High-value token identification - numbers, proper nouns, technical terms
2. Context preservation - include just enough context for each high-value token
3. Structured output template - guide LLM to output in high-overlap format
4. Aggressive deduplication - no redundant tokens
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

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*(?:g|kg|kJ|kcal|%|mg|ml|L))?")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

CONTEXT_CEILING = 400
OUTPUT_CEILING = 90

SYSTEM_PROMPT = (
    "You are Vocabulary Laser. Given high-value facts, produce a structured bullet "
    "summary with exact numbers, brand names, ingredients, and scores. Format: "
    "use bullets with labels like Brand:, Product:, Energy:, etc. "
    "Include units. Be precise and complete."
)


class LaserFact(BaseModel):
    """Fact optimized for vocabulary overlap."""
    text: str
    tokens: int = Field(..., ge=1)
    high_value_count: int = Field(..., ge=0)
    numeric_count: int = Field(..., ge=0)
    proper_noun_count: int = Field(..., ge=0)
    laser_score: float = Field(..., ge=0.0)
    category: str


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_vocabulary_laser(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    facts = _extract_laser_facts(doc)
    
    if not facts:
        raise ValueError("Vocabulary Laser: no facts extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_facts(facts, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, selected, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.1,
    )
    
    usage = response.usage.model_dump()
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "facts_extracted": len(facts),
        "facts_selected": len(selected),
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


def _extract_laser_facts(doc: DocumentView) -> list[LaserFact]:
    """Extract facts optimized for vocabulary overlap."""
    if doc.payload is not None:
        return _extract_json_laser(doc.payload)
    return _extract_text_laser(doc.raw_text)


def _extract_json_laser(payload: Any, path: tuple[str, ...] = (), depth: int = 0) -> list[LaserFact]:
    """Extract high-value facts from JSON."""
    if depth > 6:
        return []
    
    facts: list[LaserFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:80]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                facts.extend(_extract_json_laser(value, new_path, depth + 1))
            else:
                fact = _make_laser_fact(new_path, value)
                if fact:
                    facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:20]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_laser(item, new_path, depth + 1))
            else:
                fact = _make_laser_fact(new_path, item)
                if fact:
                    facts.append(fact)
    
    return facts


def _categorize_key(key: str) -> str:
    """Categorize JSON key for prioritization."""
    key_lower = key.lower()
    
    high_priority = ["brand", "name", "product", "code", "energy", "fat", "protein", 
                     "carbohydrate", "sugar", "salt", "fiber", "score", "grade",
                     "nutriscore", "ecoscore", "allergen", "ingredient", "country",
                     "serving", "quantity", "nova", "calories", "kcal"]
    
    for term in high_priority:
        if term in key_lower:
            return "priority"
    
    if "nutriment" in key_lower or "nutrition" in key_lower:
        return "nutrition"
    
    return "standard"


def _make_laser_fact(path: tuple[str, ...], value: Any) -> LaserFact | None:
    """Create laser-focused fact with high-value token analysis."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    
    formatted = _format_value(value)
    if not formatted or len(formatted) < 1:
        return None
    
    label = ".".join(path[-2:]) if len(path) > 1 else (path[0] if path else "value")
    text = f"{label}: {formatted}"
    
    tokens = len(text.split())
    if tokens < 1 or tokens > 30:
        return None
    
    numbers = NUMBER_RE.findall(text)
    proper_nouns = PROPER_NOUN_RE.findall(text)
    
    numeric_count = len(numbers)
    proper_noun_count = len(proper_nouns)
    
    category = _categorize_key(label)
    category_bonus = 3.0 if category == "priority" else (1.5 if category == "nutrition" else 0.5)
    
    high_value = numeric_count + proper_noun_count
    laser_score = (high_value * 2.0 + category_bonus + numeric_count * 1.5) / max(1, tokens)
    
    return LaserFact(
        text=text,
        tokens=tokens,
        high_value_count=high_value,
        numeric_count=numeric_count,
        proper_noun_count=proper_noun_count,
        laser_score=laser_score,
        category=category,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().split())[:150]
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)
    return str(value)[:150] if value else ""


def _extract_text_laser(text: str) -> list[LaserFact]:
    """Extract high-value sentences from text."""
    sentences = _split_sentences(text)
    facts: list[LaserFact] = []
    
    for sentence in sentences:
        clean = " ".join(sentence.split())
        
        tokens = len(clean.split())
        if tokens < 4 or tokens > 40:
            continue
        
        numbers = NUMBER_RE.findall(clean)
        proper_nouns = PROPER_NOUN_RE.findall(clean)
        
        numeric_count = len(numbers)
        proper_noun_count = len(proper_nouns)
        high_value = numeric_count + proper_noun_count
        
        if high_value == 0:
            continue
        
        laser_score = (high_value * 2.0 + numeric_count * 1.5) / tokens
        
        facts.append(LaserFact(
            text=clean,
            tokens=tokens,
            high_value_count=high_value,
            numeric_count=numeric_count,
            proper_noun_count=proper_noun_count,
            laser_score=laser_score,
            category="text",
        ))
    
    return facts


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [s.strip() for s in parts if s.strip()]


def _select_facts(facts: list[LaserFact], budget: int) -> list[LaserFact]:
    """Select facts by laser score and category diversity."""
    priority_facts = [f for f in facts if f.category == "priority"]
    nutrition_facts = [f for f in facts if f.category == "nutrition"]
    other_facts = [f for f in facts if f.category not in ("priority", "nutrition")]
    
    priority_facts.sort(key=lambda f: -f.laser_score)
    nutrition_facts.sort(key=lambda f: -f.laser_score)
    other_facts.sort(key=lambda f: -f.laser_score)
    
    selected: list[LaserFact] = []
    tokens_used = 0
    seen: set[str] = set()
    
    for fact in priority_facts[:15]:
        if fact.text.lower() in seen:
            continue
        if tokens_used + fact.tokens > budget:
            continue
        selected.append(fact)
        seen.add(fact.text.lower())
        tokens_used += fact.tokens
    
    for fact in nutrition_facts[:12]:
        if fact.text.lower() in seen:
            continue
        if tokens_used + fact.tokens > budget:
            continue
        selected.append(fact)
        seen.add(fact.text.lower())
        tokens_used += fact.tokens
    
    for fact in other_facts[:10]:
        if fact.text.lower() in seen:
            continue
        if tokens_used + fact.tokens > budget:
            continue
        selected.append(fact)
        seen.add(fact.text.lower())
        tokens_used += fact.tokens
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 200
    base = math.log(word_count + 20, 1.6) * 100
    return int(max(190, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 60
    base = math.sqrt(word_count) * 2.0
    return int(max(55, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, facts: list[LaserFact], output_budget: int) -> str:
    """Build prompt emphasizing high-value vocabulary."""
    lines = [
        f"Summarize: {doc.goal}",
        f"File: {Path(doc.path).name}",
        "",
        "Key facts (include ALL numbers, brands, scores in your summary):",
        "",
    ]
    
    for fact in facts:
        lines.append(f"• {fact.text}")
    
    lines.extend([
        "",
        f"Output {output_budget} tokens max. Use bullet format with labels.",
        "Include: brand, product name, nutritional values with units, scores/grades.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_vocabulary_laser()
    print(summary.model_dump())

