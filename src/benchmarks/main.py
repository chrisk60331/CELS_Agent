from __future__ import annotations

from typing import Any, Sequence

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands_tools import editor, file_read, file_write

from benchmarks.agent_run_summary import AgentRunSummary
from constants import MODEL_ID, task

# Define a focused system prompt for file operations
FILE_SYSTEM_PROMPT = """You are a file operations specialist. You help users read, 
write, search, and modify files. Focus on providing clear information about file 
operations and always confirm when files have been modified.

Key Capabilities:
1. Read files with various options (full content, line ranges, search)
2. Create and write to files
3. Edit existing files with precision
4. Report file information and statistics

Always specify the full file path in your responses for clarity.
"""


def run_default_agent(task_override: str | None = None) -> AgentRunSummary:
    """Execute the baseline strands agent and capture its usage metrics."""
    goal = (task_override or task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; populate constants.task.")
    file_agent = Agent(
        system_prompt=FILE_SYSTEM_PROMPT,
        tools=[file_read, file_write, editor],
        model=MODEL_ID
    )

    response = file_agent(goal)
    final_answer = _extract_response_text(response)
    metrics_obj = getattr(response, "metrics", None)
    accumulated = getattr(metrics_obj, "accumulated_usage", None) if metrics_obj is not None else None

    return AgentRunSummary.from_usage(accumulated, metadata={"final_answer": final_answer})


def _extract_response_text(result: AgentResult) -> str:
    message = getattr(result, "message", None)
    if not isinstance(message, dict):
        raise ValueError("Agent result is missing a structured message payload.")

    content_blocks: Sequence[dict[str, Any]] | None = message.get("content")
    if not isinstance(content_blocks, Sequence):
        raise ValueError("Agent message payload does not contain a content sequence.")

    texts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    if not texts:
        raise ValueError("Agent response did not include textual content.")

    return "\n\n".join(texts)


if __name__ == "__main__":
    summary = run_default_agent()
    print(summary.usage or None)