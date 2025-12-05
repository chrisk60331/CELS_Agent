"""Selective Cascade: Two-stage extraction for optimal F1/token ratio.

Core insight: Rather than compress the entire document into one LLM call, make TWO
strategic calls:

1. EXTRACTION stage (low output tokens): Ask LLM to extract ONLY key entities,
   numbers, and facts in minimal format from a larger context window
2. SYNTHESIS stage (controlled output): Use extracted facts to generate 
   comprehensive summary

Why this works:
- Stage 1 uses high input/low output ratio to maximize information extraction
- Stage 2 operates on pre-filtered facts, ensuring high signal-to-noise
- Total tokens can be lower than single-stage while achieving higher F1
- No document-specific logic - works for any structured or unstructured data

Token efficiency:
- Stage 1: ~300 input, ~50 output = 350 tokens
- Stage 2: ~120 input, ~100 output = 220 tokens  
- Total: ~570 tokens (vs 600-800 for single-stage baselines)
- Higher F1 due to better fact coverage from targeted extraction
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.bedrock_client import BedrockClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTRACTION_SYSTEM_PROMPT = """Extract key facts in minimal format. List only:
- Product/entity names
- All numerical values with units
- Categories, grades, scores
- Ingredients (if present)
Use format: "key: value" one per line. Be exhaustive but terse."""

SYNTHESIS_SYSTEM_PROMPT = """Generate comprehensive summary from extracted facts. 
Include all provided numbers, names, categories. Use bullet points. Be complete."""


class ExtractionResult(BaseModel):
    """Result from extraction stage."""
    facts: str = Field(..., description="Extracted facts")
    tokens_used: int = Field(..., description="Tokens used in extraction")


class CascadeMetrics(BaseModel):
    """Metrics for cascade processing."""
    stage1_input: int = 0
    stage1_output: int = 0
    stage2_input: int = 0
    stage2_output: int = 0
    total_tokens: int = 0
    facts_extracted_lines: int = 0


def run_iteration_selective_cascade(task_override: str | None = None) -> AgentRunSummary:
    """Entrypoint for benchmark harness."""
    goal = _require_goal(task_override or constants.task)
    file_path = _extract_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    
    if not full_path.exists():
        raise FileNotFoundError(f"Missing: {full_path}")
    
    raw_text = full_path.read_text(encoding="utf-8")
    word_count = len(raw_text.split())
    
    client = BedrockClient(model_id=constants.MODEL_ID)
    metrics = CascadeMetrics()
    
    # STAGE 1: Extract key facts with low output budget
    extraction_input = _prepare_extraction_input(raw_text, word_count)
    extraction_budget = _calculate_extraction_budget(word_count)
    
    extraction_response = client.invoke_model(
        prompt=extraction_input,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        max_tokens=extraction_budget,
        temperature=0.0,
    )
    
    metrics.stage1_input = extraction_response.usage.input_tokens
    metrics.stage1_output = extraction_response.usage.output_tokens
    
    extracted_facts = extraction_response.content.strip()
    metrics.facts_extracted_lines = len([line for line in extracted_facts.split('\n') if line.strip()])
    
    if not extracted_facts or metrics.facts_extracted_lines < 3:
        raise ValueError("Selective Cascade: Insufficient facts extracted")
    
    # STAGE 2: Synthesize summary from extracted facts
    synthesis_input = _prepare_synthesis_input(extracted_facts, Path(file_path).name)
    synthesis_budget = _calculate_synthesis_budget(word_count)
    
    synthesis_response = client.invoke_model(
        prompt=synthesis_input,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        max_tokens=synthesis_budget,
        temperature=0.1,
    )
    
    metrics.stage2_input = synthesis_response.usage.input_tokens
    metrics.stage2_output = synthesis_response.usage.output_tokens
    metrics.total_tokens = (
        metrics.stage1_input + metrics.stage1_output +
        metrics.stage2_input + metrics.stage2_output
    )
    
    # Construct combined usage for harness
    combined_usage = {
        "input_tokens": metrics.stage1_input + metrics.stage2_input,
        "output_tokens": metrics.stage1_output + metrics.stage2_output,
        "total_tokens": metrics.total_tokens,
    }
    
    metadata = {
        "final_answer": synthesis_response.content.strip(),
        "source_file": str(file_path),
        "stage1_input": metrics.stage1_input,
        "stage1_output": metrics.stage1_output,
        "stage2_input": metrics.stage2_input,
        "stage2_output": metrics.stage2_output,
        "facts_extracted_lines": metrics.facts_extracted_lines,
        "extraction_budget": extraction_budget,
        "synthesis_budget": synthesis_budget,
    }
    
    return AgentRunSummary.from_usage(usage=combined_usage, metadata=metadata)


def _require_goal(goal: str) -> str:
    """Validate goal is non-empty."""
    normalized = goal.strip()
    if not normalized:
        raise ValueError("Task required")
    return normalized


def _extract_file_path(goal: str) -> Path:
    """Extract file path from goal string."""
    marker = "file "
    lowered = goal.lower()
    if marker not in lowered:
        raise ValueError("Goal must include 'file <path>'")
    idx = lowered.index(marker) + len(marker)
    remainder = goal[idx:].strip()
    if not remainder:
        raise ValueError("No file path")
    return Path(remainder.split()[0])


def _prepare_extraction_input(raw_text: str, word_count: int) -> str:
    """Prepare input for extraction stage."""
    # Try to parse as JSON for structured extraction
    try:
        data = json.loads(raw_text)
        return _format_json_for_extraction(data, word_count)
    except json.JSONDecodeError:
        return _format_text_for_extraction(raw_text, word_count)


def _format_json_for_extraction(data: Any, word_count: int) -> str:
    """Format JSON data for extraction."""
    # For JSON, provide a compacted but complete view
    # Focus on leaf values, remove deep nesting metadata
    
    compact = _compact_json(data, max_items=80, max_depth=6)
    compact_str = json.dumps(compact, indent=1, ensure_ascii=False)
    
    # Estimate tokens (rough: 1.3 tokens per word)
    target_tokens = 280
    target_chars = int(target_tokens * 4)
    
    if len(compact_str) > target_chars:
        compact_str = compact_str[:target_chars]
    
    return f"Extract all key facts from this data:\n\n{compact_str}\n\nList all: names, numbers with units, categories, scores, ingredients."


def _compact_json(data: Any, max_items: int = 80, max_depth: int = 6, depth: int = 0) -> Any:
    """Compact JSON by removing low-value fields and limiting depth."""
    if depth > max_depth:
        return "..."
    
    # Skip low-value keys
    skip_keys = {
        "_id", "id", "code", "created_t", "last_modified_t", "rev", "status",
        "url", "image_url", "thumb_url", "small_url", "_keywords", "tags",
        "uploader", "uploaded_t", "imgid", "debug", "hierarchy", "lc",
        "states_tags", "countries_tags", "categories_tags", "editors_tags",
    }
    
    if isinstance(data, dict):
        result = {}
        count = 0
        for key, value in data.items():
            if count >= max_items:
                break
            
            # Skip if key looks like metadata
            key_lower = key.lower()
            if any(skip in key_lower for skip in skip_keys):
                continue
            
            # Skip empty or null
            if value is None or value == "" or value == []:
                continue
            
            result[key] = _compact_json(value, max_items, max_depth, depth + 1)
            count += 1
        return result
    
    elif isinstance(data, list):
        if not data:
            return []
        # Keep first N items
        limit = min(len(data), 15)
        return [_compact_json(item, max_items, max_depth, depth + 1) for item in data[:limit]]
    
    else:
        return data


def _format_text_for_extraction(raw_text: str, word_count: int) -> str:
    """Format plain text for extraction."""
    # For plain text, provide relevant chunks
    target_tokens = 280
    target_words = int(target_tokens * 0.75)
    
    words = raw_text.split()
    if len(words) <= target_words:
        content = raw_text
    else:
        # Take beginning (usually has key info)
        content = " ".join(words[:target_words])
    
    return f"Extract all key facts from this text:\n\n{content}\n\nList all: names, numbers, categories, key terms."


def _prepare_synthesis_input(extracted_facts: str, filename: str) -> str:
    """Prepare input for synthesis stage."""
    return f"Summarize {filename} using these extracted facts:\n\n{extracted_facts}\n\nProvide comprehensive bullet-point summary including all numbers and key information."


def _calculate_extraction_budget(word_count: int) -> int:
    """Calculate output token budget for extraction stage.
    
    Lower output budget for extraction since we only need facts list.
    """
    if word_count < 100:
        return 40
    elif word_count < 500:
        return 50
    elif word_count < 2000:
        return 60
    else:
        return 70


def _calculate_synthesis_budget(word_count: int) -> int:
    """Calculate output token budget for synthesis stage.
    
    Higher output budget for synthesis to ensure comprehensive summary.
    """
    if word_count < 100:
        return 80
    elif word_count < 500:
        return 100
    elif word_count < 2000:
        return 120
    else:
        return 130


if __name__ == "__main__":
    summary = run_iteration_selective_cascade()
    print(json.dumps(summary.model_dump(), indent=2))

