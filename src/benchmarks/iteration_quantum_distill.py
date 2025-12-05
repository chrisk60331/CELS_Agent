"""Quantum Distill: Deterministic fact extraction + minimal LLM formatting.

Core insight: The best F1/token comes from extracting facts directly from the 
document (for vocabulary overlap) and using LLM minimally for formatting only.
This "quantum" approach collapses the document into discrete facts before 
measurement (LLM call).

Key innovations:
1. Pre-extract exact vocabulary - identify high-value terms directly
2. Deterministic fact selection - no LLM randomness in what to include
3. Template-guided generation - LLM just fills in a structure
4. Ultra-minimal prompt - absolute minimum tokens to context
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
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

HIGH_VALUE_KEYS = {
    "brand", "brands", "product_name", "name", "code", "_id",
    "energy_100g", "energy", "fat_100g", "fat", "carbohydrates_100g", 
    "carbohydrates", "sugars_100g", "sugars", "proteins_100g", "proteins",
    "protein_100g", "protein", "salt_100g", "salt", "fiber_100g", "fiber",
    "grade", "score", "nutriscore_grade", "ecoscore_grade", "nova_group",
    "allergens", "countries", "serving_size", "quantity", "categories",
    "ingredients_text", "labels"
}

CONTEXT_CEILING = 320
OUTPUT_CEILING = 100

SYSTEM_PROMPT = (
    "Format these facts as a bullet summary. Keep exact numbers and names. "
    "Use the provided vocabulary. Be concise."
)


class QuantumFact(BaseModel):
    """Directly extracted fact with source vocabulary."""
    key: str
    value: str
    priority: int = Field(..., ge=0)


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_quantum_distill(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    facts = _extract_quantum_facts(doc)
    
    if not facts:
        raise ValueError("Quantum Distill: no facts extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_facts(facts, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, selected, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.0,
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


def _extract_quantum_facts(doc: DocumentView) -> list[QuantumFact]:
    """Deterministically extract high-value facts."""
    if doc.payload is not None:
        return _extract_json_quantum(doc.payload)
    return _extract_text_quantum(doc.raw_text)


def _is_high_value_key(key: str) -> bool:
    """Check if key is high-value."""
    key_lower = key.lower()
    for hvk in HIGH_VALUE_KEYS:
        if hvk in key_lower:
            return True
    return False


def _extract_json_quantum(payload: Any, path: tuple[str, ...] = (), depth: int = 0) -> list[QuantumFact]:
    """Extract facts from JSON targeting high-value keys."""
    if depth > 4:
        return []
    
    facts: list[QuantumFact] = []
    
    if isinstance(payload, dict):
        for key, value in payload.items():
            new_path = path + (key,)
            
            if isinstance(value, (dict, list)):
                facts.extend(_extract_json_quantum(value, new_path, depth + 1))
            else:
                is_high_value = _is_high_value_key(key)
                if is_high_value or depth <= 1:
                    fact = _make_quantum_fact(new_path, key, value)
                    if fact:
                        facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:5]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_quantum(item, new_path, depth + 1))
    
    return facts


def _make_quantum_fact(path: tuple[str, ...], key: str, value: Any) -> QuantumFact | None:
    """Create quantum fact with deterministic priority."""
    if value is None:
        return None
    
    if isinstance(value, str):
        formatted = value.strip()
        if not formatted or len(formatted) > 120:
            return None
        if formatted.startswith("http"):
            return None
    elif isinstance(value, bool):
        formatted = "Yes" if value else "No"
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            formatted = str(int(value))
        else:
            formatted = str(value)
    else:
        return None
    
    key_lower = key.lower()
    
    priority = 0
    if "brand" in key_lower:
        priority = 100
    elif "product_name" in key_lower or key_lower == "name":
        priority = 95
    elif "energy" in key_lower:
        priority = 90
    elif key_lower in ("fat_100g", "fat"):
        priority = 88
    elif "carbohydrate" in key_lower:
        priority = 87
    elif "sugar" in key_lower:
        priority = 86
    elif "protein" in key_lower:
        priority = 85
    elif "salt" in key_lower:
        priority = 84
    elif "fiber" in key_lower:
        priority = 83
    elif "grade" in key_lower or "nutriscore" in key_lower:
        priority = 80
    elif "ecoscore" in key_lower:
        priority = 78
    elif "nova" in key_lower:
        priority = 76
    elif "allergen" in key_lower:
        priority = 75
    elif "country" in key_lower or "countries" in key_lower:
        priority = 70
    elif "serving" in key_lower or "quantity" in key_lower:
        priority = 68
    elif "ingredient" in key_lower:
        priority = 65
    elif "label" in key_lower:
        priority = 62
    elif "categories" in key_lower:
        priority = 60
    elif "code" in key_lower or key_lower == "_id":
        priority = 55
    elif any(c.isdigit() for c in formatted):
        priority = 40
    else:
        priority = 20
    
    return QuantumFact(key=key, value=formatted, priority=priority)


def _extract_text_quantum(text: str) -> list[QuantumFact]:
    """Extract facts from plain text."""
    facts: list[QuantumFact] = []
    
    kv_pattern = re.compile(r"([A-Za-z][A-Za-z\s]{1,25}):\s*([^\n]{2,80})")
    
    for match in kv_pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        
        if not key or not value:
            continue
        
        priority = 50
        if any(c.isdigit() for c in value):
            priority = 60
        
        facts.append(QuantumFact(key=key, value=value, priority=priority))
    
    return facts


def _select_facts(facts: list[QuantumFact], budget: int) -> list[QuantumFact]:
    """Select facts by priority within budget."""
    ranked = sorted(facts, key=lambda f: -f.priority)
    
    selected: list[QuantumFact] = []
    tokens_used = 0
    seen_keys: set[str] = set()
    
    for fact in ranked:
        key_lower = fact.key.lower()
        if key_lower in seen_keys:
            continue
        
        line = f"{fact.key}: {fact.value}"
        line_tokens = len(line.split())
        
        if tokens_used + line_tokens > budget:
            continue
        
        if len(selected) >= 35:
            break
        
        selected.append(fact)
        seen_keys.add(key_lower)
        tokens_used += line_tokens
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 180
    base = math.log(word_count + 20, 1.7) * 80
    return int(max(170, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 65
    base = math.sqrt(word_count) * 2.2
    return int(max(60, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, facts: list[QuantumFact], output_budget: int) -> str:
    """Build ultra-minimal prompt from quantum facts."""
    lines = [f"Summary of {Path(doc.path).name}:", ""]
    
    for fact in facts:
        lines.append(f"{fact.key}: {fact.value}")
    
    lines.extend(["", f"Format as {output_budget}-token bullets. Keep all values."])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_quantum_distill()
    print(summary.model_dump())

