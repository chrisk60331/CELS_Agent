"""Benchmark runner that applies LLMLingua prompt compression."""
from __future__ import annotations

from typing import Any, Dict

from strands import Agent
from strands_tools import editor, file_read, file_write

from src.agent_run_summary import AgentRunSummary
from benchmarks.main import FILE_SYSTEM_PROMPT, _extract_response_text
from constants import MODEL_ID, task

LLMLINGUA_RATE = 0.45
LLMLINGUA_DYNAMIC_CONTEXT_RATIO = 0.35
LLMLINGUA_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
LLMLINGUA_USE_LLMLINGUA2 = True
_LLMLINGUA_COMPRESSOR: Any | None = None


def run_llmlingua_agent(task_override: str | None = None) -> AgentRunSummary:
    """Run the default file agent with an LLMLingua-compressed goal."""
    goal = (task_override or task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; populate constants.task.")

    compressed_goal, compression_meta = _compress_goal(goal)
    system_prompt = _build_system_prompt()

    file_agent = Agent(
        system_prompt=system_prompt,
        tools=[file_read, file_write, editor],
        model=MODEL_ID,
    )

    response = file_agent(compressed_goal)
    final_answer = _extract_response_text(response)

    metrics_obj = getattr(response, "metrics", None)
    accumulated = getattr(metrics_obj, "accumulated_usage", None) if metrics_obj is not None else None

    metadata = {
        "final_answer": final_answer,
        "llmlingua": compression_meta,
    }
    return AgentRunSummary.from_usage(accumulated, metadata=metadata)


def _build_system_prompt() -> str:
    return (
        FILE_SYSTEM_PROMPT.strip()
        + "\\n\\n"
        + (
            "Compression policy: the user goal has been compressed with LLMLingua. "
            "Explain if compression alters fidelity and double-check any inferred file paths."
        )
    )


def _compress_goal(goal: str) -> tuple[str, Dict[str, float | str]]:
    compressor = _get_llmlingua_compressor()
    prompt_segments = [goal]

    compression = compressor.compress_prompt(
        context=prompt_segments,
        question=goal,
        rate=LLMLINGUA_RATE,
        reorder_context="sort",
        condition_in_question="after_condition",
        dynamic_context_compression_ratio=LLMLINGUA_DYNAMIC_CONTEXT_RATIO,
    )

    compressed_goal = compression.get("compressed_prompt") if isinstance(compression, dict) else None
    if not isinstance(compressed_goal, str) or not compressed_goal.strip():
        compressed_goal = goal

    original_chars = sum(len(segment) for segment in prompt_segments)
    compressed_chars = len(compressed_goal)
    compression_ratio = compressed_chars / max(1, original_chars)

    meta: Dict[str, float | str] = {
        "compression_ratio": round(compression_ratio, 4),
        "original_chars": float(original_chars),
        "compressed_chars": float(compressed_chars),
        "configured_rate": LLMLINGUA_RATE,
    }

    if isinstance(compression, dict):
        maybe_rate = compression.get("rate") or compression.get("target_rate")
        if isinstance(maybe_rate, (int, float)):
            meta["achieved_rate"] = float(maybe_rate)

    model_name = getattr(compressor, "model_name", None)
    if isinstance(model_name, str):
        meta["model_name"] = model_name

    return compressed_goal.strip(), meta


def _get_llmlingua_compressor() -> Any:
    global _LLMLINGUA_COMPRESSOR
    if _LLMLINGUA_COMPRESSOR is not None:
        return _LLMLINGUA_COMPRESSOR
    try:
        from llmlingua import PromptCompressor
        try:
            import torch
        except Exception:  # pragma: no cover - torch may not be compiled
            torch = None
    except ImportError as exc:  # pragma: no cover - dependency missing is a runtime issue
        raise RuntimeError(
            "LLMLingua is required for this benchmark. Install it with `pip install llmlingua`."
        ) from exc

    device_map = "cuda"
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        device_map = "cpu"

    compressor_kwargs = {
        "model_name": LLMLINGUA_MODEL_NAME,
        "device_map": device_map,
    }
    if LLMLINGUA_USE_LLMLINGUA2:
        compressor_kwargs["use_llmlingua2"] = True

    _LLMLINGUA_COMPRESSOR = PromptCompressor(**compressor_kwargs)
    return _LLMLINGUA_COMPRESSOR


if __name__ == "__main__":  # pragma: no cover - manual smoke run helper
    summary = run_llmlingua_agent()
    print(summary.usage or None)
