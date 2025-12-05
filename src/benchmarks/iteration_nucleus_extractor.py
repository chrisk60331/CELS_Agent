"""Nucleus Extractor: Extract the essential nucleus of document information.

Core insight: Every document has a "nucleus" - the minimal set of facts that 
define its identity (product names, key numbers, categories). Extract ONLY 
this nucleus, format it densely, and maximize output token utilization.

Key innovations:
1. Identity extraction - find the document's defining attributes
2. Numerical completeness - capture ALL significant numbers with context
3. Categorical coverage - include grades, scores, labels, categories
4. Dense formatting - pack maximum vocabulary into output tokens
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEASUREMENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(g|kg|kJ|kcal|%|mg|ml|L|kj|cal)?", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

IDENTITY_KEYS = frozenset({
    "name", "product_name", "brand", "brands", "code", "_id", "id",
    "product_name_en", "generic_name"
})

NUMERIC_KEYS = frozenset({
    "energy", "fat", "saturated", "carbohydrates", "sugars", "protein",
    "salt", "fiber", "sodium", "calories", "kcal", "kj", "100g"
})

CATEGORY_KEYS = frozenset({
    "grade", "score", "nutriscore", "ecoscore", "nova", "categories",
    "allergens", "labels", "countries", "ingredients", "serving"
})

CONTEXT_CEILING = 450
OUTPUT_CEILING = 120

SYSTEM_PROMPT = (
    "Summarize concisely. Use bullet points. Include: product name, brand, "
    "nutritional values (energy kJ/kcal, fat, carbs, protein, salt, sugar, fiber), "
    "scores (Nutriscore, Ecoscore, NOVA), allergens, key ingredients with %. "
    "Include all numbers with units. Be complete and precise."
)


class NucleusFact(BaseModel):
    """Core fact in document nucleus."""
    text: str
    fact_type: str  # identity, numeric, category, other
    priority: float = Field(..., ge=0.0)


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_nucleus_extractor(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    nucleus = _extract_nucleus(doc)
    
    if not nucleus:
        raise ValueError("Nucleus Extractor: no nucleus facts extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_nucleus(nucleus, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, selected, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.12,
    )
    
    usage = response.usage.model_dump()
    
    type_counts = defaultdict(int)
    for fact in selected:
        type_counts[fact.fact_type] += 1
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "nucleus_extracted": len(nucleus),
        "nucleus_selected": len(selected),
        "type_distribution": dict(type_counts),
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


def _classify_key(key: str) -> str:
    """Classify key into fact type."""
    key_lower = key.lower()
    
    for ik in IDENTITY_KEYS:
        if ik in key_lower:
            return "identity"
    
    for nk in NUMERIC_KEYS:
        if nk in key_lower:
            return "numeric"
    
    for ck in CATEGORY_KEYS:
        if ck in key_lower:
            return "category"
    
    return "other"


def _extract_nucleus(doc: DocumentView) -> list[NucleusFact]:
    """Extract nucleus facts from document."""
    if doc.payload is not None:
        return _extract_json_nucleus(doc.payload)
    return _extract_text_nucleus(doc.raw_text)


def _extract_json_nucleus(payload: Any, path: tuple[str, ...] = (), depth: int = 0) -> list[NucleusFact]:
    """Extract nucleus from JSON."""
    if depth > 5:
        return []
    
    facts: list[NucleusFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:90]:
            new_path = path + (key,)
            if isinstance(value, dict):
                facts.extend(_extract_json_nucleus(value, new_path, depth + 1))
            elif isinstance(value, list):
                if value and not isinstance(value[0], (dict, list)):
                    fact = _make_nucleus_fact(new_path, value[:5])
                    if fact:
                        facts.append(fact)
                else:
                    facts.extend(_extract_json_nucleus(value, new_path, depth + 1))
            else:
                fact = _make_nucleus_fact(new_path, value)
                if fact:
                    facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:10]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_nucleus(item, new_path, depth + 1))
    
    return facts


def _make_nucleus_fact(path: tuple[str, ...], value: Any) -> NucleusFact | None:
    """Create nucleus fact from path and value."""
    if value is None:
        return None
    
    if isinstance(value, list):
        str_values = [str(v) for v in value if v is not None]
        if not str_values:
            return None
        formatted = ", ".join(str_values)
    elif isinstance(value, str):
        formatted = value.strip()
        if not formatted or len(formatted) > 150:
            return None
    elif isinstance(value, bool):
        formatted = "Yes" if value else "No"
    elif isinstance(value, (int, float)):
        if isinstance(value, float):
            if value == int(value):
                formatted = str(int(value))
            else:
                formatted = f"{value:.2f}".rstrip('0').rstrip('.')
        else:
            formatted = str(value)
    else:
        return None
    
    if not formatted:
        return None
    
    key = path[-1] if path else "value"
    key_clean = key.strip("[]0123456789_")
    
    fact_type = _classify_key(key_clean)
    full_path_str = ".".join(str(p).strip("[]") for p in path[-3:] if p and not p.startswith("["))
    
    text = f"{full_path_str}: {formatted}" if full_path_str else formatted
    
    has_number = bool(MEASUREMENT_RE.search(formatted))
    
    priority = 0.0
    if fact_type == "identity":
        priority = 10.0
    elif fact_type == "numeric":
        priority = 8.0
    elif fact_type == "category":
        priority = 6.0
    
    if has_number:
        priority += 3.0
    
    if priority < 3.0:
        return None
    
    return NucleusFact(text=text, fact_type=fact_type, priority=priority)


def _extract_text_nucleus(text: str) -> list[NucleusFact]:
    """Extract nucleus from plain text."""
    facts: list[NucleusFact] = []
    
    kv_pattern = re.compile(r"([A-Za-z][A-Za-z\s]{1,30}):\s*([^\n]{2,80})")
    
    for match in kv_pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        
        if not key or not value:
            continue
        
        fact_type = _classify_key(key)
        has_number = bool(MEASUREMENT_RE.search(value))
        
        priority = 0.0
        if fact_type == "identity":
            priority = 10.0
        elif fact_type == "numeric":
            priority = 8.0
        elif fact_type == "category":
            priority = 6.0
        
        if has_number:
            priority += 3.0
        
        if priority >= 3.0:
            facts.append(NucleusFact(
                text=f"{key}: {value}",
                fact_type=fact_type,
                priority=priority,
            ))
    
    return facts


def _select_nucleus(facts: list[NucleusFact], budget: int) -> list[NucleusFact]:
    """Select nucleus facts ensuring type diversity."""
    by_type: dict[str, list[NucleusFact]] = defaultdict(list)
    for fact in facts:
        by_type[fact.fact_type].append(fact)
    
    for ft in by_type:
        by_type[ft].sort(key=lambda f: -f.priority)
    
    selected: list[NucleusFact] = []
    tokens_used = 0
    seen: set[str] = set()
    
    type_quotas = {
        "identity": 8,
        "numeric": 20,
        "category": 10,
        "other": 5,
    }
    
    for fact_type in ["identity", "numeric", "category", "other"]:
        quota = type_quotas.get(fact_type, 5)
        for fact in by_type.get(fact_type, [])[:quota]:
            text_lower = fact.text.lower()
            if text_lower in seen:
                continue
            
            tokens = len(fact.text.split())
            if tokens_used + tokens > budget:
                continue
            
            selected.append(fact)
            seen.add(text_lower)
            tokens_used += tokens
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 220
    base = math.log(word_count + 15, 1.55) * 105
    return int(max(200, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 80
    base = math.sqrt(word_count) * 2.6
    return int(max(75, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, facts: list[NucleusFact], output_budget: int) -> str:
    """Build prompt from nucleus facts."""
    identity_facts = [f for f in facts if f.fact_type == "identity"]
    numeric_facts = [f for f in facts if f.fact_type == "numeric"]
    category_facts = [f for f in facts if f.fact_type == "category"]
    other_facts = [f for f in facts if f.fact_type == "other"]
    
    lines = [
        f"Summarize {Path(doc.path).name}",
        "",
    ]
    
    if identity_facts:
        lines.append("Identity:")
        for f in identity_facts:
            lines.append(f"  {f.text}")
    
    if numeric_facts:
        lines.append("Nutritional data:")
        for f in numeric_facts:
            lines.append(f"  {f.text}")
    
    if category_facts:
        lines.append("Categories/scores:")
        for f in category_facts:
            lines.append(f"  {f.text}")
    
    if other_facts:
        lines.append("Other:")
        for f in other_facts:
            lines.append(f"  {f.text}")
    
    lines.extend([
        "",
        f"Generate {output_budget}-token structured bullet summary.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_nucleus_extractor()
    print(summary.model_dump())

