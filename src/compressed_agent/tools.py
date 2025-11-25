"""Tool interface for reading/writing latent substrate."""
from typing import Dict, Any, Optional, List, Set
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from .file_summarizer import summarize_data_file
from .latent_state import LatentState, GraphNode, GraphEdge, Factor, MicroSummary


class ToolResult(BaseModel):
    """Result from a tool execution."""
    success: bool = Field(..., description="Whether the tool execution succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    state_edits: List[Dict[str, Any]] = Field(default_factory=list, description="Proposed state edits")
    error: Optional[str] = Field(None, description="Error message if failed")


class Tool(ABC):
    """Base class for tools that read/write latent substrate."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(
        self,
        state: LatentState,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute the tool and return result with state edits."""
        pass

    def read_state(self, state: LatentState, node_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Read relevant parts of the state (minimal token usage)."""
        if node_ids is None:
            # Return compressed summary
            return {
                "node_count": len(state.nodes),
                "edge_count": len(state.edges),
                "factor_count": len(state.factors),
                "summary_count": len(state.summaries),
                "version": state.version
            }

        # Return only requested nodes and their immediate context
        result = {
            "nodes": {},
            "edges": [],
            "factors": [],
            "summaries": []
        }

        for node_id in node_ids:
            if node_id in state.nodes:
                node = state.nodes[node_id]
                result["nodes"][node_id] = {
                    "type": node.type.value,
                    "label": node.label,
                    "properties": node.properties
                }

                # Include connected edges
                for edge in state.get_edges_from(node_id):
                    result["edges"].append({
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type.value
                    })

                # Include relevant factors
                for factor in state.get_factors_for_node(node_id):
                    result["factors"].append({
                        "id": factor.id,
                        "name": factor.name,
                        "constraint_type": factor.constraint_type
                    })

        # Include relevant summaries
        summaries = state.get_summaries_for_scope(node_ids)
        for summary in summaries:
            result["summaries"].append({
                "id": summary.id,
                "summary": summary.summary,
                "scope": list(summary.scope)
            })

        return result


class QueryTool(Tool):
    """Tool for querying the latent state."""

    def __init__(self):
        super().__init__(
            name="query_state",
            description="Query the latent state for nodes, edges, or factors"
        )

    def execute(
        self,
        state: LatentState,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Query the state."""
        query_type = parameters.get("type", "node")
        query_value = parameters.get("value")

        if query_type == "node":
            if query_value in state.nodes:
                node = state.nodes[query_value]
                return ToolResult(
                    success=True,
                    data={
                        "node": {
                            "id": node.id,
                            "type": node.type.value,
                            "label": node.label,
                            "properties": node.properties
                        }
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Node {query_value} not found"
                )

        elif query_type == "path":
            # Find path between nodes
            source = parameters.get("source")
            target = parameters.get("target")
            if source and target:
                path = self._find_path(state, source, target)
                return ToolResult(
                    success=True,
                    data={"path": path}
                )

        return ToolResult(
            success=False,
            error=f"Unknown query type: {query_type}"
        )

    def _find_path(self, state: LatentState, source: str, target: str) -> List[str]:
        """Simple BFS path finding."""
        from collections import deque
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()
            if current == target:
                return path

            for edge in state.get_edges_from(current):
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))

        return []


class CreateNodeTool(Tool):
    """Tool for creating nodes in the latent state."""

    def __init__(self):
        super().__init__(
            name="create_node",
            description="Create a new node in the latent state"
        )

    def execute(
        self,
        state: LatentState,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Create a node."""
        from .state_edit import StateEdit
        from .latent_state import NodeType

        node_type = NodeType(parameters.get("type", "entity"))
        label = parameters.get("label", "unnamed")
        properties = parameters.get("properties", {})

        edit = StateEdit(
            operation="add_node",
            data={
                "type": node_type.value,
                "label": label,
                "properties": properties
            },
            reason=f"Tool {self.name} created node",
            priority=3
        )

        return ToolResult(
            success=True,
            data={"node_id": edit.data.get("id")},
            state_edits=[edit.dict()]
        )


class CreateEdgeTool(Tool):
    """Tool for creating edges in the latent state."""

    def __init__(self):
        super().__init__(
            name="create_edge",
            description="Create a new edge in the latent state"
        )

    def execute(
        self,
        state: LatentState,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Create an edge."""
        from .state_edit import StateEdit
        from .latent_state import EdgeType

        source = parameters.get("source")
        target = parameters.get("target")
        edge_type = EdgeType(parameters.get("type", "depends_on"))
        weight = parameters.get("weight", 1.0)

        if not source or not target:
            return ToolResult(
                success=False,
                error="Source and target are required"
            )

        edit = StateEdit(
            operation="add_edge",
            data={
                "source": source,
                "target": target,
                "type": edge_type.value,
                "weight": weight
            },
            reason=f"Tool {self.name} created edge",
            priority=3
        )

        return ToolResult(
            success=True,
            data={"edge_id": f"{source}_{target}_{edge_type.value}"},
            state_edits=[edit.dict()]
        )


class CreateSummaryTool(Tool):
    """Tool for creating micro-summaries."""

    def __init__(self):
        super().__init__(
            name="create_summary",
            description="Create a micro-summary for a set of nodes"
        )

    def execute(
        self,
        state: LatentState,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Create a summary."""
        from .state_edit import StateEdit

        scope = set(parameters.get("scope", []))
        summary_text = parameters.get("summary", "")

        if not scope or not summary_text:
            return ToolResult(
                success=False,
                error="Scope and summary text are required"
            )

        edit = StateEdit(
            operation="add_summary",
            data={
                "scope": list(scope),
                "summary": summary_text
            },
            reason=f"Tool {self.name} created summary",
            priority=2
        )

        return ToolResult(
            success=True,
            data={"summary_id": f"summary_{len(state.summaries)}"},
            state_edits=[edit.dict()]
        )


class FileReadTool(Tool):
    """Tool for reading files and storing content in latent state."""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read a file and store its content in the latent state"
        )

    def execute(
        self,
        state: LatentState,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Read a file and create nodes/summaries from its content."""
        from .state_edit import StateEdit

        file_path = parameters.get("path")
        if not file_path:
            return ToolResult(
                success=False,
                error="File path is required"
            )

        try:
            # Resolve path relative to project root
            project_root = Path(__file__).parent.parent.parent
            full_path = (project_root / file_path).resolve()
            
            if not full_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )

            # Read file content
            content = full_path.read_text(encoding="utf-8")

            summary_text = summarize_data_file(full_path, content)

            # Create a summary node for the file content
            edit = StateEdit(
                operation="add_summary",
                data={
                    "scope": [str(full_path)],
                    "summary": summary_text[:1000]
                },
                reason=f"Read file {file_path}",
                priority=5
            )

            return ToolResult(
                success=True,
                data={
                    "file_path": str(full_path),
                    "content_length": len(content),
                    "content_preview": content[:200],
                    "summary": summary_text
                },
                state_edits=[edit.dict()]
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error reading file: {str(e)}"
            )


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        self.register(QueryTool())
        self.register(CreateNodeTool())
        self.register(CreateEdgeTool())
        self.register(CreateSummaryTool())
        self.register(FileReadTool())

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]

