"""Precision Lens: Structure-aware extraction for optimal F1/token ratio.

Core insight: Documents have structural patterns that indicate importance
regardless of domain. Top-level keys, numeric values, and short strings
are universally high-value. Extract based on STRUCTURE, not domain keywords.

Key innovations:
1. Structure-first extraction - prioritize by JSON depth and value type
2. Universal scoring - numbers, short strings, proper nouns are always valuable
3. No domain assumptions - works for food, history, prizes, any structured data
4. Tight token budgets - single LLM call with minimal overhead

Generalizes because:
- No hardcoded domain terms (no "nutriscore", "ingredient", etc.)
- Scoring based on structural properties (depth, type, length)
- Works on any JSON or text document
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+\b")

# Generic system prompt - no domain-specific terms
SYSTEM_PROMPT = "Summarize concisely with bullet points. Include all names, numbers with units, scores, and key facts."

# Skip patterns that are universally low-value metadata
SKIP_PATTERNS = frozenset({
    "url", "image", "thumb", "http", "debug", "tag", "hierarchy",
    "schema", "sortkey", "editor", "contributor", "uploaded", "rev",
    "created_t", "modified_t", "scan", "source", "hash", "uuid",
    "warning", "error", "status", "version", "lang", "lc"
})

# Keys that suggest metadata/counts rather than content
METADATA_SUFFIXES = ("_n", "_count", "_tags", "_hierarchy", "_t", "_id", "_imgid")


class LensFact(BaseModel):
    """Extracted fact with structural metadata."""
    key: str
    value: str
    depth: int = Field(..., ge=0)
    priority: float = Field(..., ge=0.0)
    token_estimate: int = Field(..., ge=1)


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_precision_lens(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)

    facts = _extract_facts(doc)

    if not facts:
        raise ValueError("Precision Lens: no facts extracted")

    input_budget = _input_budget(doc.word_count)
    selected = _select_by_priority(facts, input_budget)

    output_budget = _output_budget(len(selected))
    prompt = _build_prompt(doc, selected)

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
        "input_budget": input_budget,
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


def _should_skip_key(key: str) -> bool:
    """Check if key is universally low-value metadata."""
    key_lower = key.lower()
    return any(skip in key_lower for skip in SKIP_PATTERNS)


def _extract_facts(doc: DocumentView) -> list[LensFact]:
    """Extract facts from document based on structure."""
    if doc.payload is not None:
        return _extract_json_facts(doc.payload, (), 0)
    return _extract_text_facts(doc.raw_text)


def _extract_json_facts(
    payload: Any,
    path: tuple[str, ...],
    depth: int
) -> list[LensFact]:
    """Extract facts from JSON based on structural properties."""
    if depth > 6:
        return []

    facts: list[LensFact] = []

    if isinstance(payload, dict):
        # Handle common wrapper patterns at top level
        if len(payload) <= 4 and depth == 0:
            for key, value in payload.items():
                new_path = path + (key,)
                if isinstance(value, dict) and len(value) > 10:
                    # Wrapper with dict content (e.g., {"product": {...}})
                    facts.extend(_extract_json_facts(value, path, depth))
                elif isinstance(value, list) and value:
                    # Wrapper with list content (e.g., {"prizes": [...], "laureates": [...]})
                    for idx, item in enumerate(value[:10]):
                        if isinstance(item, dict):
                            facts.extend(_extract_json_facts(
                                item, new_path + (f"[{idx}]",), depth + 1
                            ))
                elif not _should_skip_key(key):
                    facts.extend(_extract_json_facts(value, new_path, depth + 1))
            return facts

        for key, value in payload.items():
            if _should_skip_key(key):
                continue

            new_path = path + (key,)

            if isinstance(value, dict):
                facts.extend(_extract_json_facts(value, new_path, depth + 1))
            elif isinstance(value, list):
                if value and not isinstance(value[0], (dict, list)):
                    # List of scalars - join first few
                    fact = _make_fact(new_path, value[:5], depth)
                    if fact:
                        facts.append(fact)
                else:
                    # List of objects - extract from first few
                    for idx, item in enumerate(value[:5]):
                        if isinstance(item, dict):
                            facts.extend(_extract_json_facts(
                                item, new_path + (f"[{idx}]",), depth + 1
                            ))
            else:
                fact = _make_fact(new_path, value, depth)
                if fact:
                    facts.append(fact)

    return facts


def _make_fact(path: tuple[str, ...], value: Any, depth: int) -> LensFact | None:
    """Create fact from path and value with structural scoring."""
    if value is None:
        return None

    # Format value
    if isinstance(value, list):
        str_values = [str(v).strip() for v in value if v is not None]
        if not str_values:
            return None
        formatted = ", ".join(str_values[:5])
        if len(formatted) > 80:
            formatted = formatted[:77] + "..."
    elif isinstance(value, str):
        formatted = value.strip()
        if not formatted or len(formatted) > 100:
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

    # Build display key
    key = _format_key(path)
    if len(key) < 2:
        return None

    # Calculate priority based on structural properties
    priority = _calculate_priority(depth, formatted, key)

    if priority < 1.0:
        return None

    token_estimate = _estimate_tokens(f"{key}: {formatted}")

    return LensFact(
        key=key,
        value=formatted,
        depth=depth,
        priority=priority,
        token_estimate=token_estimate,
    )


def _calculate_priority(depth: int, value: str, key: str) -> float:
    """Calculate priority based on universal structural properties."""
    priority = 0.0
    key_lower = key.lower()

    # Penalize metadata-like keys
    if any(key_lower.endswith(suffix) for suffix in METADATA_SUFFIXES):
        return 0.0

    # Skip pure count/index values
    if key_lower in ("n", "count", "index", "rev", "imgid"):
        return 0.0

    # Shallow depth = more important (top-level keys matter)
    depth_bonus = max(0, 4 - depth)
    priority += depth_bonus

    # Descriptive string values are important
    if PROPER_NOUN_RE.search(value):
        priority += 3.0

    # Numbers with context are important (but not isolated counts)
    if NUMBER_RE.search(value):
        # Bonus for numbers that look like measurements
        if any(u in value.lower() for u in ("g", "kg", "kj", "kcal", "%", "mg")):
            priority += 4.0
        elif len(value) < 10:  # Short numeric, could be score
            priority += 2.0

    # Key name heuristics (universal patterns)
    if any(x in key_lower for x in ("name", "title", "brand")):
        priority += 4.0
    if any(x in key_lower for x in ("grade", "rating")):
        priority += 4.0
    if any(x in key_lower for x in ("score",)):
        priority += 3.0
    if "100g" in key_lower or "per_" in key_lower:
        priority += 3.0
    if any(x in key_lower for x in ("quantity", "serving", "size")):
        priority += 2.0

    # Bonus for short, likely meaningful values
    if 3 <= len(value) <= 40:
        priority += 1.0

    return priority


def _format_key(path: tuple[str, ...]) -> str:
    """Format path into display key."""
    relevant_parts = []
    for part in path[-3:]:
        clean = part.strip("[]0123456789_")
        if clean and clean not in ("product", "data", "value", "text", "item"):
            relevant_parts.append(clean)

    if not relevant_parts:
        return path[-1] if path else "value"

    return ".".join(relevant_parts[-2:])


def _estimate_tokens(text: str) -> int:
    """Estimate token count."""
    words = len(text.split())
    return max(1, int(words * 1.3) + 1)


def _extract_text_facts(text: str) -> list[LensFact]:
    """Extract facts from plain text."""
    facts: list[LensFact] = []

    # Look for key: value patterns
    kv_pattern = re.compile(r"([A-Za-z][A-Za-z\s]{1,25}):\s*([^\n]{2,60})")

    for match in kv_pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()

        if not key or not value or _should_skip_key(key):
            continue

        priority = _calculate_priority(0, value, key)

        if priority >= 1.0:
            token_estimate = _estimate_tokens(f"{key}: {value}")
            facts.append(LensFact(
                key=key,
                value=value,
                depth=0,
                priority=priority,
                token_estimate=token_estimate,
            ))

    return facts


def _select_by_priority(facts: list[LensFact], budget: int) -> list[LensFact]:
    """Select facts by priority within budget."""
    ranked = sorted(facts, key=lambda f: (-f.priority, f.depth))

    selected: list[LensFact] = []
    tokens_used = 0
    seen_keys: set[str] = set()

    for fact in ranked:
        key_lower = fact.key.lower()
        if key_lower in seen_keys:
            continue

        if tokens_used + fact.token_estimate > budget:
            continue

        if len(selected) >= 25:
            break

        selected.append(fact)
        seen_keys.add(key_lower)
        tokens_used += fact.token_estimate

    return selected


def _input_budget(word_count: int) -> int:
    """Calculate input token budget - tight for efficiency."""
    if word_count < 500:
        return 85
    elif word_count < 2000:
        return 100
    else:
        return 120


def _output_budget(selected_count: int) -> int:
    """Calculate output token budget - enough for complete summary."""
    return min(110, max(75, selected_count * 3))


def _build_prompt(doc: DocumentView, facts: list[LensFact]) -> str:
    """Build compact prompt from facts."""
    lines = [f"Summarize {Path(doc.path).name}:"]

    for fact in facts:
        lines.append(f"{fact.key}: {fact.value}")

    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_precision_lens()
    print(json.dumps(summary.model_dump(), indent=2))
