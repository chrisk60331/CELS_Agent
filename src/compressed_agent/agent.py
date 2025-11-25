"""Agent orchestrator with multi-turn planning."""
from typing import List, Dict, Any, Optional, Tuple
import re

from pydantic import BaseModel, Field

from .bedrock_client import BedrockClient, TokenUsage
from .latent_state import LatentState
from .state_edit import StateEditPipeline, StateEdit
from .tools import ToolRegistry, Tool, ToolResult


class AgentStep(BaseModel):
    """A single step in agent execution."""
    step_id: int = Field(..., description="Step identifier")
    tool_name: str = Field(..., description="Tool used")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    result: ToolResult = Field(..., description="Tool result")
    state_edits: List[StateEdit] = Field(default_factory=list, description="State edits from this step")


class AgentPlan(BaseModel):
    """A plan for achieving a goal."""
    goal: str = Field(..., description="Goal to achieve")
    steps: List[AgentStep] = Field(default_factory=list, description="Planned steps")
    current_step: int = Field(default=0, description="Current step index")


class CompressedAgent:
    """Agent that uses compressed latent state for planning."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        bedrock_client: Optional[BedrockClient] = None
    ):
        self.state = LatentState()
        self.bedrock_client = bedrock_client
        self.pipeline = StateEditPipeline(bedrock_client=bedrock_client)
        self.tool_registry = tool_registry or ToolRegistry()
        self.history: List[AgentStep] = []
        self.token_usage = TokenUsage()

    def execute_goal(
        self,
        goal: str,
        max_steps: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a goal using compressed state edits."""
        context = context or {}
        steps_executed = 0

        # Normalize current state
        self.state = self.pipeline.normalizer.normalize(self.state)

        # Propose initial edits based on goal
        initial_edits = self.pipeline.proposer.propose_edits(self.state, goal, context)
        
        # Track token usage from proposal
        proposal_usage = self.pipeline.proposer.get_last_usage()
        if proposal_usage:
            self.token_usage.input_tokens += proposal_usage.input_tokens
            self.token_usage.output_tokens += proposal_usage.output_tokens
            self.token_usage.total_tokens += proposal_usage.total_tokens
        
        # Apply initial edits
        if initial_edits:
            self.state = self.pipeline.applier.apply_edits(self.state, initial_edits)

        # Execute tool-based steps
        while steps_executed < max_steps:
            # Select next tool and parameters based on current state and goal
            tool, parameters = self._select_tool(goal, context)

            if not tool:
                break

            # Execute tool
            result = tool.execute(self.state, parameters)

            # Extract state edits from tool result
            state_edits = []
            for edit_dict in result.state_edits:
                state_edits.append(StateEdit(**edit_dict))

            # Apply state edits
            if state_edits:
                self.state = self.pipeline.applier.apply_edits(self.state, state_edits)

            # Record step
            step = AgentStep(
                step_id=steps_executed,
                tool_name=tool.name,
                parameters=parameters,
                result=result,
                state_edits=state_edits
            )
            self.history.append(step)

            steps_executed += 1

            # Check if goal is achieved
            if self._goal_achieved(goal):
                break

        return {
            "goal": goal,
            "steps_executed": steps_executed,
            "final_state": self._compress_state_for_output(),
            "history": [step.dict() for step in self.history[-5:]],  # Last 5 steps
            "token_usage": {
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
                "total_tokens": self.token_usage.total_tokens
            }
        }

    def _select_tool(self, goal: str, context: Dict[str, Any]) -> Tuple[Optional[Tool], Dict[str, Any]]:
        """Select next tool to execute (simplified selection logic)."""
        # Simple heuristic: if goal mentions "create" or "add", use create_node
        goal_lower = goal.lower()

        if "pick" in goal_lower and "number" in goal_lower:
            return None, {}

        # Check for file reading tasks (summaries, reads, inspections)
        if self._goal_requests_file_summary(goal_lower):
            tool = self.tool_registry.get("read_file")
            if tool:
                file_path = self._extract_goal_file_path(goal) or context.get("file_path")
                if file_path:
                    return tool, {"path": file_path}

        if "create" in goal_lower or "add" in goal_lower:
            tool = self.tool_registry.get("create_node")
            if tool:
                # Extract entity name from goal
                words = goal.split()
                label = words[-1] if words else "entity"
                return tool, {"type": "entity", "label": label}

        if "connect" in goal_lower or "link" in goal_lower:
            tool = self.tool_registry.get("create_edge")
            if tool:
                # This is simplified - in practice would parse goal more carefully
                return tool, {"source": "entity_1", "target": "entity_2", "type": "depends_on"}

        if "query" in goal_lower or "find" in goal_lower:
            tool = self.tool_registry.get("query_state")
            if tool:
                return tool, {"type": "node", "value": "entity_1"}

        # Default: query state
        tool = self.tool_registry.get("query_state")
        return tool, {"type": "node", "value": list(self.state.nodes.keys())[0] if self.state.nodes else None}

    @staticmethod
    def _goal_requests_file_summary(goal_lower: str) -> bool:
        summary_keywords = ("summarize", "summary", "read", "analyze", "inspect")
        return "file" in goal_lower and any(keyword in goal_lower for keyword in summary_keywords)

    @staticmethod
    def _extract_goal_file_path(goal: str) -> Optional[str]:
        file_pattern = r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]+"
        for match in re.findall(file_pattern, goal):
            candidate = match.strip("'\" ")
            if candidate:
                return candidate
        return None

    def _goal_achieved(self, goal: str) -> bool:
        """Check if goal is achieved (simplified)."""
        goal_lower = goal.lower()
        if self._goal_requests_file_summary(goal_lower):
            return any(step.tool_name == "read_file" and step.result.success for step in self.history)
        if "create" in goal_lower or "add" in goal_lower:
            return len(self.state.nodes) > 0
        if "pick" in goal_lower and "number" in goal_lower:
            return True
        return False

    def _compress_state_for_output(self) -> Dict[str, Any]:
        """Compress state for output (minimal token usage)."""
        return {
            "node_count": len(self.state.nodes),
            "edge_count": len(self.state.edges),
            "factor_count": len(self.state.factors),
            "summary_count": len(self.state.summaries),
            "version": self.state.version,
            "nodes": {
                node_id: {"type": node.type.value, "label": node.label}
                for node_id, node in list(self.state.nodes.items())[:10]  # Limit to 10
            }
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a compressed summary of current state."""
        return self._compress_state_for_output()

    def reset(self):
        """Reset agent state."""
        self.state = LatentState()
        self.history = []
        self.token_usage = TokenUsage()
        if self.bedrock_client:
            self.bedrock_client.reset_usage()

