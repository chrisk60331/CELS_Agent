"""Sparse Primer: Ultra-minimal prompt with maximum information density.

Core insight: LLMs are pre-trained on vast corpora and already "know" how to 
summarize. Instead of verbose prompts with lots of context, provide only the 
ESSENTIAL "primer" - key identifiers and constraints - and let the LLM's 
inherent capabilities do the work with minimal token overhead.

Key innovations:
1. Minimal context - only key identifiers (names, codes, numbers)
2. Sparse evidence - just enough facts to prime the LLM
3. Tight constraints - strict output limits
4. Trust LLM knowledge - let it fill in structural templates
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

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

CONTEXT_CEILING = 280
OUTPUT_CEILING = 110

SYSTEM_PROMPT = (
    "Expert summarizer. Dense bullets. Include all numbers with units, "
    "brand names, grades, percentages. No speculation."
)


class PrimerFact(BaseModel):
    """Ultra-compact fact for priming."""
    key: str
    value: str
    priority: float = Field(..., ge=0.0)


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_sparse_primer(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    primers = _extract_primers(doc)
    
    if not primers:
        raise ValueError("Sparse Primer: no primers extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_primers(primers, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, selected, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.15,
    )
    
    usage = response.usage.model_dump()
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "primers_extracted": len(primers),
        "primers_selected": len(selected),
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


PRIORITY_KEYS = frozenset({
    "brand", "brands", "name", "product_name", "code", "energy", "fat", 
    "saturated", "carbohydrates", "sugars", "protein", "proteins", "salt",
    "fiber", "fibre", "score", "grade", "nutriscore", "ecoscore", "nova",
    "allergens", "allergen", "ingredients", "countries", "country",
    "serving", "quantity", "kcal", "calories", "categories"
})


def _extract_primers(doc: DocumentView) -> list[PrimerFact]:
    """Extract minimal key-value primers."""
    if doc.payload is not None:
        return _extract_json_primers(doc.payload)
    return _extract_text_primers(doc.raw_text)


def _extract_json_primers(payload: Any, path: tuple[str, ...] = (), depth: int = 0) -> list[PrimerFact]:
    """Extract key-value primers from JSON."""
    if depth > 5:
        return []
    
    primers: list[PrimerFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:100]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                primers.extend(_extract_json_primers(value, new_path, depth + 1))
            else:
                primer = _make_primer(new_path, value)
                if primer:
                    primers.append(primer)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:8]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                primers.extend(_extract_json_primers(item, new_path, depth + 1))
    
    return primers


def _is_priority_key(path: tuple[str, ...]) -> bool:
    """Check if path contains priority key."""
    for part in path:
        part_lower = part.lower().strip("[]0123456789")
        if part_lower in PRIORITY_KEYS:
            return True
        for pk in PRIORITY_KEYS:
            if pk in part_lower:
                return True
    return False


def _make_primer(path: tuple[str, ...], value: Any) -> PrimerFact | None:
    """Create compact primer from path and value."""
    if value is None:
        return None
    
    if isinstance(value, str):
        if not value.strip() or len(value.strip()) > 100:
            return None
        formatted = value.strip()
    elif isinstance(value, bool):
        formatted = "yes" if value else "no"
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            formatted = str(int(value))
        else:
            formatted = str(value)
    else:
        return None
    
    key = path[-1] if path else "value"
    key = key.strip("[]0123456789_")
    
    if not key or len(key) < 2:
        return None
    
    is_priority = _is_priority_key(path)
    has_number = bool(NUMBER_RE.search(formatted))
    
    priority = 0.0
    if is_priority:
        priority += 5.0
    if has_number:
        priority += 3.0
    if len(formatted) < 20:
        priority += 1.0
    
    if priority < 2.0:
        return None
    
    return PrimerFact(key=key, value=formatted, priority=priority)


def _extract_text_primers(text: str) -> list[PrimerFact]:
    """Extract primers from text using pattern matching."""
    primers: list[PrimerFact] = []
    
    colon_pattern = re.compile(r"([A-Za-z][A-Za-z\s]{2,25}):\s*([^\n]{2,60})")
    
    for match in colon_pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        
        if not key or not value:
            continue
        
        has_number = bool(NUMBER_RE.search(value))
        priority = 3.0 if has_number else 1.0
        
        if any(pk in key.lower() for pk in PRIORITY_KEYS):
            priority += 4.0
        
        if priority >= 2.0:
            primers.append(PrimerFact(key=key, value=value, priority=priority))
    
    return primers


def _select_primers(primers: list[PrimerFact], budget: int) -> list[PrimerFact]:
    """Select primers by priority within budget."""
    ranked = sorted(primers, key=lambda p: -p.priority)
    
    selected: list[PrimerFact] = []
    tokens_used = 0
    seen_keys: set[str] = set()
    
    for primer in ranked:
        key_lower = primer.key.lower()
        if key_lower in seen_keys:
            continue
        
        line_tokens = len(f"{primer.key}: {primer.value}".split())
        
        if tokens_used + line_tokens > budget:
            continue
        
        if len(selected) >= 40:
            break
        
        selected.append(primer)
        seen_keys.add(key_lower)
        tokens_used += line_tokens
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 150
    base = math.log(word_count + 25, 1.8) * 70
    return int(max(140, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 70
    base = math.sqrt(word_count) * 2.4
    return int(max(65, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, primers: list[PrimerFact], output_budget: int) -> str:
    """Build minimal primer prompt."""
    lines = [
        f"Summarize {Path(doc.path).name}:",
        "",
    ]
    
    for primer in primers:
        lines.append(f"{primer.key}: {primer.value}")
    
    lines.extend([
        "",
        f"Dense {output_budget}-token bullet summary with numbers, brands, grades.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_sparse_primer()
    print(summary.model_dump())

