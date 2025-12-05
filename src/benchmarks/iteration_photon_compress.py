"""Photon Compress: Universal ultra-efficient document compression.

Core insight: Extract the highest-value tokens from ANY document type using
generic value signals (numbers, proper nouns, short strings) rather than
domain-specific keys. Like photons carrying maximum information per quantum,
each token must be maximally informative.

Key innovations:
1. Universal value detection - works on any JSON/text structure
2. Density-first selection - maximize unique vocabulary per token
3. Ultra-tight output - strict token ceiling
4. Zero domain assumptions - no food/science/etc specific heuristics
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
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

CONTEXT_CEILING = 380
OUTPUT_CEILING = 95

SYSTEM_PROMPT = (
    "Generate a dense bullet summary from these key facts. "
    "Include all numbers, names, and important values. Be concise."
)


class PhotonFact(BaseModel):
    """Minimal fact with universal value scoring."""
    text: str
    tokens: int = Field(..., ge=1)
    density_score: float = Field(..., ge=0.0)


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_photon_compress(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    facts = _extract_photon_facts(doc)
    
    if not facts:
        raise ValueError("Photon Compress: no facts extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_by_density(facts, context_budget)
    
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


def _extract_photon_facts(doc: DocumentView) -> list[PhotonFact]:
    """Extract facts using universal value signals."""
    if doc.payload is not None:
        return _extract_json_photons(doc.payload)
    return _extract_text_photons(doc.raw_text)


def _compute_density_score(key: str, value: str) -> float:
    """Universal density scoring - no domain assumptions."""
    score = 0.0
    
    has_number = bool(NUMBER_RE.search(value))
    if has_number:
        score += 3.0
    
    tokens = TOKEN_RE.findall(value)
    unique_tokens = set(t.lower() for t in tokens if len(t) > 2)
    if unique_tokens:
        score += len(unique_tokens) * 0.5
    
    capital_words = sum(1 for t in tokens if t[0].isupper() and t.isalpha())
    if capital_words:
        score += capital_words * 0.8
    
    value_len = len(value)
    if 5 <= value_len <= 50:
        score += 1.5
    elif value_len < 5:
        score += 0.3
    
    key_len = len(key)
    if key_len <= 15:
        score += 0.5
    
    return score


def _extract_json_photons(payload: Any, path: tuple[str, ...] = (), depth: int = 0) -> list[PhotonFact]:
    """Extract facts from JSON with universal scoring."""
    if depth > 5:
        return []
    
    facts: list[PhotonFact] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:80]:
            new_path = path + (key,)
            
            if isinstance(value, dict):
                facts.extend(_extract_json_photons(value, new_path, depth + 1))
            elif isinstance(value, list):
                if len(value) > 0 and not isinstance(value[0], (dict, list)):
                    list_str = ", ".join(str(v) for v in value[:5])
                    if len(list_str) <= 100:
                        fact = _make_photon_fact(key, f"[{list_str}]")
                        if fact:
                            facts.append(fact)
                else:
                    for idx, item in enumerate(value[:6]):
                        if isinstance(item, (dict, list)):
                            facts.extend(_extract_json_photons(item, new_path + (f"[{idx}]",), depth + 1))
            else:
                fact = _make_photon_fact(key, value)
                if fact:
                    facts.append(fact)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:8]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                facts.extend(_extract_json_photons(item, new_path, depth + 1))
    
    return facts


def _make_photon_fact(key: str, value: Any) -> PhotonFact | None:
    """Create photon fact with universal scoring."""
    if value is None:
        return None
    
    if isinstance(value, str):
        formatted = value.strip()
        if not formatted or len(formatted) > 150:
            return None
        if formatted.startswith("http") or formatted.startswith("\\u"):
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
    
    text = f"{key}: {formatted}"
    tokens = len(text.split())
    
    if tokens < 2 or tokens > 25:
        return None
    
    density_score = _compute_density_score(key, formatted)
    
    return PhotonFact(text=text, tokens=tokens, density_score=density_score)


def _extract_text_photons(text: str) -> list[PhotonFact]:
    """Extract facts from plain text."""
    facts: list[PhotonFact] = []
    
    kv_pattern = re.compile(r"([A-Za-z][A-Za-z\s]{1,20}):\s*([^\n]{2,80})")
    
    for match in kv_pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        
        if not key or not value:
            continue
        
        fact = _make_photon_fact(key, value)
        if fact:
            facts.append(fact)
    
    return facts


def _select_by_density(facts: list[PhotonFact], budget: int) -> list[PhotonFact]:
    """Select facts by density score per token."""
    ranked = sorted(facts, key=lambda f: (-f.density_score / max(1, f.tokens), -f.density_score))
    
    selected: list[PhotonFact] = []
    tokens_used = 0
    seen: set[str] = set()
    
    for fact in ranked:
        key = fact.text.split(":")[0].lower() if ":" in fact.text else fact.text.lower()
        if key in seen:
            continue
        
        if tokens_used + fact.tokens > budget:
            continue
        
        if len(selected) >= 40:
            break
        
        selected.append(fact)
        seen.add(key)
        tokens_used += fact.tokens
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 200
    base = math.log(word_count + 15, 1.6) * 95
    return int(max(190, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 60
    base = math.sqrt(word_count) * 2.0
    return int(max(55, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, facts: list[PhotonFact], output_budget: int) -> str:
    """Build minimal prompt."""
    lines = [f"Summarize {Path(doc.path).name}:", ""]
    
    for fact in facts:
        lines.append(f"• {fact.text}")
    
    lines.extend(["", f"Generate {output_budget}-token bullet summary."])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_photon_compress()
    print(summary.model_dump())

