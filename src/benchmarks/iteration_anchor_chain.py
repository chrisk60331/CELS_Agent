"""Anchor Chain: Extract anchor concepts and their relational chains.

Core insight: Documents have "anchor" concepts - high-gravity nodes around which 
information clusters. By identifying anchors and their immediate relational chains,
we capture maximum information with minimal redundancy.

Key innovations:
1. Anchor detection - find concepts with highest connectivity/mention density
2. Chain extraction - for each anchor, extract its immediate predicate chain
3. Chain fusion - merge overlapping chains to eliminate redundancy
4. Minimal synthesis - generate summary from fused anchor chains only
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

SKIP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "they", "to", "was", "were", "with", "this", "these", "those", "will", "can",
    "could", "may", "might", "about", "there", "here", "than", "then", "also",
    "such", "very", "only", "some", "any", "been", "being", "would", "should",
    "what", "when", "where", "which", "who", "how", "why", "each", "other",
})

MAX_ANCHORS = 12
MAX_CHAINS_PER_ANCHOR = 4
CONTEXT_CEILING = 360
OUTPUT_CEILING = 70

SYSTEM_PROMPT = (
    "You are Anchor Chain, an information compressor that extracts core concepts "
    "and their relationship chains. Given anchor-chain evidence, synthesize a "
    "maximally dense bullet summary. Preserve numbers and names exactly. No speculation."
)


class Anchor(BaseModel):
    """Core concept anchor with connectivity score."""
    term: str
    frequency: int = Field(..., ge=1)
    connectivity: float = Field(..., ge=0.0)


class ChainLink(BaseModel):
    """A fact linking an anchor to related information."""
    anchor: str
    text: str
    tokens: int = Field(..., ge=1)
    relevance: float = Field(..., ge=0.0)
    source: str


class DocumentView(BaseModel):
    """Document representation."""
    goal: str
    path: str
    raw_text: str
    payload: Any | None = None
    word_count: int = Field(..., ge=0)


def run_iteration_anchor_chain(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    doc = _load_document(goal)
    
    anchors = _detect_anchors(doc.raw_text)
    chains = _extract_chains(doc, anchors)
    
    if not chains:
        raise ValueError("Anchor Chain: no chains extracted")
    
    context_budget = _context_budget(doc.word_count)
    selected = _select_chains(chains, context_budget)
    
    output_budget = _output_budget(doc.word_count)
    prompt = _build_prompt(doc, anchors, selected, output_budget)
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    response = client.invoke_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_budget,
        temperature=0.06,
    )
    
    usage = response.usage.model_dump()
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": doc.path,
        "anchors_detected": len(anchors),
        "chains_extracted": len(chains),
        "chains_selected": len(selected),
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


def _detect_anchors(text: str) -> list[Anchor]:
    """Detect high-gravity anchor concepts in the document."""
    tokens = [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in SKIP_WORDS and len(t) > 2]
    
    if not tokens:
        return []
    
    freq = Counter(tokens)
    
    cooccurrence: dict[str, set[str]] = defaultdict(set)
    window_size = 8
    for i, token in enumerate(tokens):
        window = tokens[max(0, i - window_size):i + window_size + 1]
        for neighbor in window:
            if neighbor != token:
                cooccurrence[token].add(neighbor)
    
    anchors = []
    for term, count in freq.most_common(MAX_ANCHORS * 3):
        connectivity = len(cooccurrence[term]) / max(1, math.log(count + 1))
        anchor_score = count * (1 + connectivity * 0.1)
        anchors.append(Anchor(term=term, frequency=count, connectivity=anchor_score))
    
    anchors.sort(key=lambda a: -a.connectivity)
    return anchors[:MAX_ANCHORS]


def _extract_chains(doc: DocumentView, anchors: list[Anchor]) -> list[ChainLink]:
    """Extract fact chains for each anchor."""
    anchor_terms = {a.term.lower() for a in anchors}
    
    if doc.payload is not None:
        return _extract_json_chains(doc.payload, anchor_terms)
    return _extract_text_chains(doc.raw_text, anchor_terms)


def _extract_json_chains(payload: Any, anchors: set[str], path: tuple[str, ...] = (), depth: int = 0) -> list[ChainLink]:
    """Extract chains from JSON structure."""
    if depth > 7:
        return []
    
    chains: list[ChainLink] = []
    
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:65]:
            new_path = path + (key,)
            if isinstance(value, (dict, list)):
                chains.extend(_extract_json_chains(value, anchors, new_path, depth + 1))
            else:
                chain = _make_chain_link(new_path, value, anchors)
                if chain:
                    chains.append(chain)
    
    elif isinstance(payload, list):
        for idx, item in enumerate(payload[:14]):
            new_path = path + (f"[{idx}]",)
            if isinstance(item, (dict, list)):
                chains.extend(_extract_json_chains(item, anchors, new_path, depth + 1))
            else:
                chain = _make_chain_link(new_path, item, anchors)
                if chain:
                    chains.append(chain)
    
    return chains


def _make_chain_link(path: tuple[str, ...], value: Any, anchors: set[str]) -> ChainLink | None:
    """Create chain link if value connects to an anchor."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    
    formatted = _format_value(value)
    if not formatted or len(formatted) < 2:
        return None
    
    label = ".".join(path) if path else "value"
    text = f"{label}: {formatted}"
    text_lower = text.lower()
    
    matching_anchors = [a for a in anchors if a in text_lower]
    if not matching_anchors:
        key_lower = label.lower()
        path_anchors = [a for a in anchors if a in key_lower]
        if not path_anchors:
            return None
        matching_anchors = path_anchors
    
    tokens = len(text.split())
    if tokens < 2 or tokens > 40:
        return None
    
    anchor_count = len(matching_anchors)
    numeric_bonus = 1.3 if any(c.isdigit() for c in formatted) else 0.3
    depth_penalty = 1.0 + len(path) * 0.1
    
    relevance = (2.0 + anchor_count * 1.5 + numeric_bonus) / depth_penalty
    
    return ChainLink(
        anchor=matching_anchors[0],
        text=text,
        tokens=tokens,
        relevance=relevance,
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


def _extract_text_chains(text: str, anchors: set[str]) -> list[ChainLink]:
    """Extract sentence chains for anchors."""
    sentences = _split_sentences(text)
    chains: list[ChainLink] = []
    
    for idx, sentence in enumerate(sentences):
        clean = " ".join(sentence.split())
        sentence_lower = clean.lower()
        
        matching_anchors = [a for a in anchors if a in sentence_lower]
        if not matching_anchors:
            continue
        
        tokens = len(clean.split())
        if tokens < 5 or tokens > 55:
            continue
        
        anchor_count = len(matching_anchors)
        numeric_bonus = 1.0 if any(c.isdigit() for c in clean) else 0.2
        position_factor = 1.0 / (1.0 + idx * 0.03)
        
        relevance = (2.5 + anchor_count * 1.2 + numeric_bonus) * position_factor
        
        chains.append(ChainLink(
            anchor=matching_anchors[0],
            text=clean,
            tokens=tokens,
            relevance=relevance,
            source="text",
        ))
    
    return chains


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [s.strip() for s in parts if s.strip()]


def _select_chains(chains: list[ChainLink], budget: int) -> list[ChainLink]:
    """Select chains balancing anchor coverage and relevance."""
    anchor_chains: dict[str, list[ChainLink]] = defaultdict(list)
    for chain in chains:
        anchor_chains[chain.anchor].append(chain)
    
    for anchor in anchor_chains:
        anchor_chains[anchor].sort(key=lambda c: -c.relevance)
    
    selected: list[ChainLink] = []
    tokens_used = 0
    seen: set[str] = set()
    
    for anchor in anchor_chains:
        for chain in anchor_chains[anchor][:MAX_CHAINS_PER_ANCHOR]:
            if chain.text.lower() in seen:
                continue
            if tokens_used + chain.tokens > budget:
                continue
            
            selected.append(chain)
            seen.add(chain.text.lower())
            tokens_used += chain.tokens
    
    remaining = sorted(chains, key=lambda c: -c.relevance / max(1, c.tokens))
    for chain in remaining:
        if chain.text.lower() in seen:
            continue
        if tokens_used + chain.tokens > budget:
            continue
        if len(selected) >= 32:
            break
        
        selected.append(chain)
        seen.add(chain.text.lower())
        tokens_used += chain.tokens
    
    return selected


def _context_budget(word_count: int) -> int:
    if word_count <= 0:
        return 180
    base = math.log(word_count + 18, 1.7) * 88
    return int(max(170, min(CONTEXT_CEILING, base)))


def _output_budget(word_count: int) -> int:
    if word_count <= 0:
        return 48
    base = math.sqrt(word_count) * 1.65
    return int(max(44, min(OUTPUT_CEILING, base)))


def _build_prompt(doc: DocumentView, anchors: list[Anchor], chains: list[ChainLink], output_budget: int) -> str:
    """Build prompt from anchor chains."""
    anchor_terms = [a.term for a in anchors[:8]]
    
    lines = [
        f"Task: {doc.goal}",
        f"Source: {Path(doc.path).name}",
        f"Anchors: {', '.join(anchor_terms)}",
        "",
        f"Synthesize a {output_budget}-token bullet summary from these anchor chains:",
        "",
    ]
    
    for chain in chains:
        lines.append(f"[{chain.anchor}] {chain.text}")
    
    lines.extend([
        "",
        "Output bullets only. Keep numbers exact. No speculation.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_iteration_anchor_chain()
    print(summary.model_dump())

