"""Graph-based knowledge extraction for optimal F1/token using centrality and connectivity."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import constants
from src.agent_run_summary import AgentRunSummary
from src.compressed_agent.file_summarizer import summarize_data_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_DEPTH = 6
MAX_NODES = 200


class KnowledgeNode(BaseModel):
    """Node in knowledge graph."""

    id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Node label")
    value: str = Field(..., description="Node value")
    depth: int = Field(..., ge=0, description="Depth in structure")
    connections: int = Field(default=0, description="Number of connections")
    centrality: float = Field(default=0.0, description="Centrality score")
    tokens: int = Field(..., ge=1, description="Token count")


class KnowledgeGraph:
    """Knowledge graph representation of document structure."""

    def __init__(self):
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[tuple[str, str]] = []
        self.node_connections: defaultdict[str, set[str]] = defaultdict(set)

    def add_node(self, node_id: str, label: str, value: str, depth: int) -> None:
        """Add node to graph."""
        tokens = max(1, _token_count(label) + _token_count(value))
        self.nodes[node_id] = KnowledgeNode(
            id=node_id, label=label, value=value, depth=depth, tokens=tokens
        )

    def add_edge(self, source_id: str, target_id: str) -> None:
        """Add edge between nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append((source_id, target_id))
            self.node_connections[source_id].add(target_id)
            self.node_connections[target_id].add(source_id)

    def compute_centrality(self) -> None:
        """Compute centrality scores for all nodes."""
        for node_id, node in self.nodes.items():
            connections = len(self.node_connections[node_id])
            node.connections = connections
            depth_penalty = 1.0 / (1.0 + node.depth * 0.2)
            centrality = connections * depth_penalty
            node.centrality = centrality

    def get_key_nodes(self, budget: int) -> list[KnowledgeNode]:
        """Get key nodes based on centrality within budget."""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: (-n.centrality, n.tokens, n.depth))
        selected: list[KnowledgeNode] = []
        tokens_used = 0
        seen_labels: set[str] = set()

        for node in sorted_nodes:
            label_key = node.label.lower()
            if label_key in seen_labels:
                continue
            if tokens_used + node.tokens > budget:
                continue
            selected.append(node)
            tokens_used += node.tokens
            seen_labels.add(label_key)
            if len(selected) >= 20:
                break

        return selected


class GraphExtractor:
    """Extracts knowledge graph from document structure."""

    def __init__(self, payload: Any | None):
        self.payload = payload
        self.graph = KnowledgeGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build knowledge graph from payload."""
        if self.payload is None:
            return
        self._walk(self.payload, (), None, 0)

    def _walk(self, value: Any, path: tuple[str, ...], parent_id: str | None, depth: int) -> None:
        """Recursively walk structure and build graph."""
        if depth > MAX_DEPTH or len(self.graph.nodes) >= MAX_NODES:
            return

        current_id = ".".join(path) if path else "root"

        if isinstance(value, dict):
            self.graph.add_node(current_id, _format_path(path), "object", depth)
            if parent_id:
                self.graph.add_edge(parent_id, current_id)

            for key, child in value.items():
                child_path = path + (key,)
                child_id = ".".join(child_path)
                self._walk(child, child_path, current_id, depth + 1)
            return

        if isinstance(value, list):
            self.graph.add_node(current_id, _format_path(path), f"list[{len(value)}]", depth)
            if parent_id:
                self.graph.add_edge(parent_id, current_id)

            for idx, child in enumerate(value[:5]):
                child_path = path + (f"[{idx}]",)
                self._walk(child, child_path, current_id, depth + 1)
            return

        scalar_text = _normalize_scalar(value)
        if scalar_text:
            self.graph.add_node(current_id, _format_path(path), scalar_text, depth)
            if parent_id:
                self.graph.add_edge(parent_id, current_id)


def run_iteration_graph_extractor(task_override: str | None = None) -> AgentRunSummary:
    """Main entrypoint for graph-based extraction."""
    goal = (task_override or constants.task or "").strip()
    if not goal:
        raise ValueError("Benchmark task is empty.")

    file_path = _extract_goal_file_path(goal)
    full_path = (PROJECT_ROOT / file_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Source file not found: {full_path}")

    raw_text = full_path.read_text(encoding="utf-8")
    payload = _try_load_json(raw_text)
    base_summary = summarize_data_file(full_path, raw_text)

    extractor = GraphExtractor(payload)
    extractor.graph.compute_centrality()
    word_count = _token_count(raw_text)
    budget = _compute_graph_budget(word_count)
    key_nodes = extractor.graph.get_key_nodes(budget)

    summary_lines = [f"## GraphExtractor Summary of `{full_path.name}`"]
    summary_lines.append(f"Centrality-based selection ({len(key_nodes)} nodes, {len(extractor.graph.edges)} edges)")

    for node in key_nodes:
        summary_lines.append(f"- {node.label}: {node.value}")

    if len(key_nodes) < 5:
        summary_lines.append("\n### Additional Context")
        base_lines = _extract_summary_lines(base_summary)
        summary_lines.extend(f"- {line}" for line in base_lines[:3])

    summary_text = "\n".join(summary_lines).strip()

    usage = _estimate_usage(raw_text, summary_text)
    metadata = {
        "final_answer": summary_text,
        "source_file": str(file_path),
        "node_count": len(key_nodes),
        "total_nodes": len(extractor.graph.nodes),
        "edge_count": len(extractor.graph.edges),
        "token_budget": budget,
    }

    return AgentRunSummary.from_usage(usage=usage, metadata=metadata)


def _extract_goal_file_path(goal: str) -> Path:
    """Extract file path from goal string."""
    markers = ["file ", "summarize file "]
    goal_lower = goal.lower()
    for marker in markers:
        if marker in goal_lower:
            start_idx = goal_lower.index(marker) + len(marker)
            remainder = goal[start_idx:].strip()
            if remainder:
                candidate = remainder.split()[0]
                return Path(candidate)
    raise ValueError(f"Unable to parse file path from goal: {goal}")


def _try_load_json(raw_text: str) -> Any | None:
    """Attempt to parse JSON."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def _normalize_scalar(value: Any) -> str:
    """Normalize scalar value to string."""
    if isinstance(value, str):
        compact = " ".join(value.strip().split())
        return compact[:160]
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _format_path(path: tuple[str, ...]) -> str:
    """Format path tuple to dot-separated string."""
    return ".".join(path) if path else "root"


def _token_count(text: str) -> int:
    """Estimate token count."""
    return max(1, len(text.split()))


def _compute_graph_budget(word_count: int) -> int:
    """Compute token budget for graph extraction."""
    if word_count < 150:
        return 65
    if word_count < 1500:
        return int(_token_count(str(word_count)) * 5 + 40)
    return int(_token_count(str(word_count)) * 6 + 50)


def _extract_summary_lines(summary_text: str) -> list[str]:
    """Extract meaningful lines from summary."""
    lines: list[str] = []
    for raw_line in summary_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("##"):
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if len(stripped) > 5:
            lines.append(stripped)
    return lines


def _estimate_usage(source_text: str, summary_text: str) -> dict[str, int]:
    """Estimate token usage."""
    input_tokens = _token_count(source_text)
    output_tokens = _token_count(summary_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


if __name__ == "__main__":
    summary = run_iteration_graph_extractor()
    print(summary.model_dump())

