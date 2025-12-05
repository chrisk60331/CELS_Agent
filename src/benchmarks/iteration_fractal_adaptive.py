"""Fractal Adaptive Compressor: Optimizes F1/token via entropy-based fractal selection.

Core insight: Information is not evenly distributed. It clusters in "high-entropy" regions.
By aggressively filtering for "Semantic Nuclei" (Ingredients, Nutrients, Classifications)
and discarding "Metadata Crust" (IDs, Dates, URLs, Logs), we maximize F1/Token.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import wordfreq

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Regex for "Entities" (Heuristic)
RE_NUMBER = re.compile(r"\d+(?:[\.,]\d+)?")
RE_CAPS = re.compile(r"\b[A-Z][a-z]+\b")
RE_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", re.IGNORECASE)
RE_HASH = re.compile(r"\b[0-9a-f]{20,}\b", re.IGNORECASE)
RE_URL = re.compile(r"http[s]?://")

# Minimal System Prompt
SYSTEM_PROMPT = "Summarize. Focus: Identity, Ingredients, Nutrition. concise bullet points."

class FractalShard(BaseModel):
    """A minimal unit of information."""
    path: str
    content: str
    entropy: float
    estimated_tokens: int
    
class DocumentFractal(BaseModel):
    """Fractal representation of the document."""
    shards: list[FractalShard] = []
    total_entropy: float = 0.0

def run_iteration_fractal_adaptive(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    file_path = _extract_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    
    if not full_path.exists():
        raise FileNotFoundError(f"Missing: {full_path}")
        
    raw_text = full_path.read_text(encoding="utf-8")
    
    # 1. Fractal Decomposition & Scoring
    fractal = _decompose_and_score(raw_text)
    
    if not fractal.shards:
        raise ValueError("Fractal Adaptive: No information extracted")

    # 2. Aggressive Budgeting
    # Target: < 350 Total Tokens.
    # Output: 120. Input: 200.
    budget_tokens = 200
    
    # 3. Selection
    selected_shards = _select_shards(fractal, budget_tokens)
    
    # 4. Prompt Engineering
    prompt_text = _construct_prompt(selected_shards)
    
    # 5. Bedrock Execution
    client = BedrockClient(model_id=constants.MODEL_ID)
    output_max = 120
    
    response = client.invoke_model(
        prompt=prompt_text,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=output_max,
        temperature=0.1,
    )
    
    usage = response.usage.model_dump()
    
    metadata = {
        "final_answer": response.content.strip(),
        "source_file": str(file_path),
        "shards_extracted": len(fractal.shards),
        "shards_selected": len(selected_shards),
        "total_entropy": fractal.total_entropy,
        "input_budget": budget_tokens,
    }
    
    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)

def _require_goal(goal: str) -> str:
    normalized = goal.strip()
    if not normalized:
        raise ValueError("Task required")
    return normalized

def _extract_file_path(goal: str) -> Path:
    parts = goal.split()
    for i, part in enumerate(parts):
        if part == "file" and i + 1 < len(parts):
            return Path(parts[i+1])
        if part.endswith(".json") or part.endswith(".txt"):
            return Path(part)
    if "file " in goal:
        return Path(goal.split("file ")[1].strip())
    raise ValueError(f"Could not extract file path from: {goal}")

def _decompose_and_score(raw_text: str) -> DocumentFractal:
    """Decomposes document into scored shards."""
    shards = []
    try:
        data = json.loads(raw_text)
        _flatten_json(data, "", shards)
    except json.JSONDecodeError:
        _flatten_text(raw_text, shards)
        
    if not shards:
        return DocumentFractal()
        
    total_entropy = sum(s.entropy for s in shards)
    return DocumentFractal(shards=shards, total_entropy=total_entropy)

def _flatten_json(data: Any, prefix: str, shards: list[FractalShard], depth: int = 0):
    if depth > 8: return
    
    if isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            _flatten_json(v, new_prefix, shards, depth + 1)
    elif isinstance(data, list):
        # Flatten first few items only
        for i, v in enumerate(data):
            if i > 15: break
            new_prefix = f"{prefix}[{i}]"
            _flatten_json(v, new_prefix, shards, depth + 1)
    else:
        content = str(data).strip()
        if content:
            entropy = _calculate_entropy(prefix, content)
            # Conservative estimation: 2.5x words
            leaf = prefix.split('.')[-1]
            path_tokens = len(leaf.split('_')) * 2
            content_tokens = len(content.split()) * 2.5
            est_tokens = int(2 + path_tokens + content_tokens)
            
            shards.append(FractalShard(
                path=prefix,
                content=content,
                entropy=entropy,
                estimated_tokens=est_tokens
            ))

def _flatten_text(text: str, shards: list[FractalShard]):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent: continue
        entropy = _calculate_entropy(f"S{i}", sent)
        est_tokens = int(len(sent.split()) * 2.5) + 2
        shards.append(FractalShard(
            path=f"line_{i}",
            content=sent,
            entropy=entropy,
            estimated_tokens=est_tokens
        ))

def _calculate_entropy(key: str, content: str) -> float:
    """Calculates 'entropy' (informativeness) of a content string."""
    if not content or content.lower() in ["null", "none", "n/a", "unknown"]:
        return 0.0
        
    key_lower = key.lower()
    
    # KILL LIST
    kill_keywords = [
        "url", "image", "thumb", "small", "full", "display", 
        "date", "time", "tstamp", "created", "modified", "updated", "uploaded",
        "rev", "version", "schema", "sortkey", "index", "page",
        "scan", "contributor", "editor", "checkers", "informers", "sources",
        "hierarchy", "debug", "error", "warning", "traces", "completeness",
        "code", "id", "uuid", "hash", "tags", "packaging", "countries", "properties"
    ]
    if any(k in key_lower for k in kill_keywords):
        return 0.0

    if RE_UUID.search(content) or RE_HASH.search(content) or RE_URL.search(content):
        return 0.0

    # BASE SCORE
    score = 1.0

    # BOOST LIST
    boost_keywords = [
        "ingredient", "nutri", "energy", "kcal", "fat", "sugar", 
        "protein", "fiber", "salt", "sodium", "carbo",
        "name", "brand", "product", "category",
        "grade", "score", "label", "allergen"
    ]
    
    if any(k in key_lower for k in boost_keywords):
        score += 15.0
        
    # Content Analysis
    if RE_NUMBER.search(content):
        score += 3.0
    if RE_CAPS.search(content):
        score += 1.0
        
    # Word Rarity
    words = re.findall(r'\w+', content.lower())
    if words:
        for w in words:
            freq = wordfreq.zipf_frequency(w, 'en')
            if freq > 0 and freq < 4.0: # Rare
                score += 2.0

    # Penalty for long text
    if len(words) > 30:
        score *= 0.3
        
    return score

def _select_shards(fractal: DocumentFractal, budget_tokens: int) -> list[FractalShard]:
    """Selects highest entropy shards."""
    scored_shards = []
    for s in fractal.shards:
        if s.entropy <= 0: continue
        efficiency = s.entropy / max(1, s.estimated_tokens)
        scored_shards.append((efficiency, s))
        
    scored_shards.sort(key=lambda x: x[0], reverse=True)
    
    selected = []
    current_tokens = 0
    seen_content = set()
    
    # Prompt Overhead reserve
    effective_budget = budget_tokens - 20
    
    for _, shard in scored_shards:
        if current_tokens + shard.estimated_tokens > effective_budget:
            continue
            
        clean_content = shard.content.lower().strip()
        if clean_content in seen_content:
            continue
            
        selected.append(shard)
        current_tokens += shard.estimated_tokens
        seen_content.add(clean_content)
        
    # Sort by path
    selected.sort(key=lambda x: x.path)
    return selected

def _construct_prompt(shards: list[FractalShard]) -> str:
    lines = []
    for s in shards:
        parts = s.path.split('.')
        key = parts[-1]
        key = re.sub(r'\[\d+\]', '', key)
        
        if key in ["text", "value", "name"] and len(parts) > 1:
            parent = parts[-2]
            parent = re.sub(r'\[\d+\]', '', parent)
            key = f"{parent}.{key}"
            
        lines.append(f"{key}: {s.content}")
        
    return "\n".join(lines)

if __name__ == "__main__":
    try:
        summary = run_iteration_fractal_adaptive()
        print(summary.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {e}")
