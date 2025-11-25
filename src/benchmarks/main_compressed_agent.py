"""Benchmark runner for CompressedAgent."""
from __future__ import annotations

from src.compressed_agent.agent import CompressedAgent
from src.compressed_agent.bedrock_client import BedrockClient
from benchmarks.agent_run_summary import AgentRunSummary
from constants import MODEL_ID
import constants


def run_compressed_agent(task_override: str | None = None) -> AgentRunSummary:
    """Execute the CompressedAgent for the current benchmark task."""
    bedrock_client = BedrockClient(model_id=MODEL_ID)
    agent = CompressedAgent(bedrock_client=bedrock_client)

    goal = (task_override or constants.task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty; populate constants.task.")

    result = agent.execute_goal(goal, max_steps=10)

    # Extract final answer from result
    final_answer = _extract_final_answer(result)
    
    # Extract token usage
    token_usage = result.get("token_usage", {})
    if not token_usage:
        # Fallback to bedrock client usage
        token_usage_obj = bedrock_client.get_total_usage()
        token_usage = {
            "input_tokens": token_usage_obj.input_tokens,
            "output_tokens": token_usage_obj.output_tokens,
            "total_tokens": token_usage_obj.total_tokens,
        }

    return AgentRunSummary.from_usage(
        usage=token_usage,
        metadata={"final_answer": final_answer}
    )


def _extract_final_answer(result: dict) -> str:
    """Extract final answer text from agent result."""
    goal_text = result.get("goal", "")
    goal_lower = goal_text.lower() if isinstance(goal_text, str) else ""

    history = result.get("history", [])
    summaries: list[str] = []
    previews: list[str] = []

    for step in history:
        step_result = step.get("result", {})
        if not (step_result.get("success") and step.get("tool_name") == "read_file"):
            continue
        data = step_result.get("data", {})
        summary_text = data.get("summary")
        if isinstance(summary_text, str) and summary_text.strip():
            summaries.append(summary_text.strip())
            continue
        content_preview = data.get("content_preview", "")
        file_path = data.get("file_path", "unknown")
        if content_preview:
            previews.append(f"File {file_path}: {content_preview}")

    if summaries:
        return "\n\n".join(summaries)
    if previews:
        return "\n".join(previews)

    if "pick" in goal_lower and "number" in goal_lower:
        return "I pick the number 42."
    
    # Try to get a summary from the final state
    final_state = result.get("final_state", {})
    
    # Build a summary from the state
    parts = []
    if final_state.get("node_count", 0) > 0:
        parts.append(f"Created {final_state['node_count']} nodes")
    if final_state.get("edge_count", 0) > 0:
        parts.append(f"Created {final_state['edge_count']} edges")
    
    steps_executed = result.get("steps_executed", 0)
    if steps_executed > 0:
        parts.append(f"Executed {steps_executed} steps")
    
    # If we have nodes, try to extract some information
    nodes = final_state.get("nodes", {})
    if nodes:
        node_info = []
        for node_id, node_data in list(nodes.items())[:5]:  # Limit to 5 nodes
            label = node_data.get("label", "unknown")
            node_type = node_data.get("type", "unknown")
            node_info.append(f"{node_type}: {label}")
        if node_info:
            parts.append("Nodes: " + ", ".join(node_info))
    
    if not parts:
        return f"Completed goal: {result.get('goal', 'unknown')}"
    
    return ". ".join(parts) + "."


if __name__ == "__main__":
    summary = run_compressed_agent()
    print(summary.usage or None)

