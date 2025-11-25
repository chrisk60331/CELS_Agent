"""State edit pipeline: normalize → propose → apply."""
from typing import List, Optional, Set, Dict, Any
from pydantic import BaseModel, Field
from .latent_state import LatentState, GraphNode, GraphEdge, Factor, MicroSummary, NodeType, EdgeType
from .bedrock_client import BedrockClient, TokenUsage
import json


class StateEdit(BaseModel):
    """A proposed edit to the latent state."""
    operation: str = Field(..., description="Operation: add_node, remove_node, add_edge, remove_edge, add_factor, remove_factor, add_summary, update_node, update_edge")
    target_id: Optional[str] = Field(None, description="Target ID for the operation")
    data: Dict[str, Any] = Field(..., description="Operation data")
    reason: str = Field(..., description="Reason for this edit")
    priority: int = Field(default=0, description="Edit priority (higher = more important)")


class StateNormalizer:
    """Normalizes the latent state to canonical form."""

    @staticmethod
    def normalize(state: LatentState) -> LatentState:
        """Normalize state to canonical form."""
        normalized = LatentState(
            nodes={},
            edges={},
            factors={},
            summaries={},
            version=state.version
        )

        # Normalize nodes: deduplicate, merge similar, canonicalize IDs
        node_mapping: Dict[str, str] = {}
        for node_id, node in state.nodes.items():
            canonical_id = StateNormalizer._canonicalize_node_id(node)
            node_mapping[node_id] = canonical_id
            if canonical_id not in normalized.nodes:
                normalized.nodes[canonical_id] = GraphNode(
                    id=canonical_id,
                    type=node.type,
                    label=node.label,
                    properties=node.properties,
                    created_at=node.created_at,
                    updated_at=node.updated_at
                )

        # Normalize edges: remap node IDs, deduplicate
        edge_set: Set[tuple] = set()
        for edge_id, edge in state.edges.items():
            source = node_mapping.get(edge.source, edge.source)
            target = node_mapping.get(edge.target, edge.target)
            edge_key = (source, target, edge.type)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                normalized.edges[edge_id] = GraphEdge(
                    id=edge_id,
                    source=source,
                    target=target,
                    type=edge.type,
                    weight=edge.weight,
                    properties=edge.properties,
                    created_at=edge.created_at
                )

        # Normalize factors: remap node IDs
        for factor_id, factor in state.factors.items():
            normalized_nodes = {node_mapping.get(n, n) for n in factor.nodes}
            normalized.factors[factor_id] = Factor(
                id=factor_id,
                name=factor.name,
                nodes=normalized_nodes,
                constraint_type=factor.constraint_type,
                value=factor.value,
                properties=factor.properties,
                created_at=factor.created_at
            )

        # Normalize summaries: remap node IDs, deduplicate by hash
        summary_hashes: Set[str] = set()
        for summary_id, summary in state.summaries.items():
            normalized_scope = {node_mapping.get(n, n) for n in summary.scope}
            if summary.hash not in summary_hashes:
                summary_hashes.add(summary.hash)
                normalized.summaries[summary_id] = MicroSummary(
                    id=summary_id,
                    scope=normalized_scope,
                    summary=summary.summary,
                    hash=summary.hash,
                    created_at=summary.created_at
                )

        normalized.version = state.version + 1
        return normalized

    @staticmethod
    def _canonicalize_node_id(node: GraphNode) -> str:
        """Generate canonical ID for a node."""
        # Simple canonicalization: use type + normalized label
        normalized_label = node.label.lower().replace(" ", "_")
        return f"{node.type.value}_{normalized_label}"


class StateProposer:
    """Proposes edits to the latent state using Bedrock."""

    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        self.bedrock_client = bedrock_client
        self.last_usage: Optional[TokenUsage] = None

    def propose_edits(
        self,
        current_state: LatentState,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[StateEdit]:
        """Propose edits to achieve a goal using Bedrock."""
        # Reset last_usage at start of each proposal
        self.last_usage = None
        
        edits: List[StateEdit] = []
        context = context or {}

        # Compress state for LLM input (minimal token usage)
        compressed_state = self._compress_state(current_state)

        # Use Bedrock if available, otherwise fallback to simple heuristic
        if self.bedrock_client:
            try:
                edits = self._propose_with_bedrock(compressed_state, goal, context)
            except Exception as e:
                # Fallback to simple heuristic if Bedrock fails
                print(f"Bedrock call failed: {e}, using fallback")
                edits = self._propose_fallback(goal)
                self.last_usage = None  # No tokens used in fallback
        else:
            edits = self._propose_fallback(goal)
            self.last_usage = None  # No tokens used in fallback

        return edits

    def _compress_state(self, state: LatentState) -> Dict[str, Any]:
        """Compress state for minimal token usage."""
        return {
            "nodes": [
                {"id": n.id, "type": n.type.value, "label": n.label}
                for n in list(state.nodes.values())[:20]  # Limit to 20 nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "type": e.type.value}
                for e in list(state.edges.values())[:20]  # Limit to 20 edges
            ],
            "factors": len(state.factors),
            "summaries": len(state.summaries),
            "version": state.version
        }

    def _propose_with_bedrock(
        self,
        compressed_state: Dict[str, Any],
        goal: str,
        context: Dict[str, Any]
    ) -> List[StateEdit]:
        """Propose edits using Bedrock."""
        # Get usage before the call to track incremental usage (copy values, not reference)
        usage_before_total = self.bedrock_client.get_total_usage()
        usage_before = TokenUsage(
            input_tokens=usage_before_total.input_tokens,
            output_tokens=usage_before_total.output_tokens,
            total_tokens=usage_before_total.total_tokens
        )
        
        system_prompt = """You are a planning agent that proposes state edits in JSON format.
Given a compressed latent state and a goal, propose minimal state edits to achieve the goal.
Return only a JSON array of edits, each with: operation, data, reason, priority.
Operations: add_node, remove_node, add_edge, remove_edge, add_factor, add_summary.
Keep edits minimal and focused."""

        prompt = f"""Current compressed state:
{json.dumps(compressed_state, indent=2)}

Goal: {goal}

Context: {json.dumps(context, indent=2)}

Propose state edits as JSON array. Example:
[
  {{
    "operation": "add_node",
    "data": {{"type": "entity", "label": "task", "properties": {{}}}},
    "reason": "Goal requires creating a task entity",
    "priority": 5
  }}
]"""

        response = self.bedrock_client.invoke_model(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.3
        )
        
        # Calculate incremental usage for this call
        usage_after = self.bedrock_client.get_total_usage()
        incremental_usage = TokenUsage(
            input_tokens=usage_after.input_tokens - usage_before.input_tokens,
            output_tokens=usage_after.output_tokens - usage_before.output_tokens,
            total_tokens=usage_after.total_tokens - usage_before.total_tokens
        )
        # Store usage BEFORE parsing (in case parsing fails)
        self.last_usage = incremental_usage

        # Parse response
        try:
            # Extract JSON from response
            content = response.content.strip()
            # Remove markdown code blocks if present
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            
            edits_data = json.loads(content)
            # Ensure it's a list
            if not isinstance(edits_data, list):
                edits_data = [edits_data]
            
            # Validate and create edits
            edits = []
            for edit_dict in edits_data:
                try:
                    # Ensure required fields exist
                    if "operation" not in edit_dict:
                        continue
                    if "data" not in edit_dict:
                        edit_dict["data"] = {}
                    if "reason" not in edit_dict:
                        edit_dict["reason"] = "Proposed by Bedrock"
                    if "priority" not in edit_dict:
                        edit_dict["priority"] = 3
                    
                    edits.append(StateEdit(**edit_dict))
                except Exception as e:
                    print(f"Failed to create edit from {edit_dict}: {e}")
                    continue
            
            return edits if edits else self._propose_fallback(goal)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback if JSON parsing fails
            print(f"Failed to parse Bedrock response: {e}")
            print(f"Response content: {response.content[:500]}")
            return self._propose_fallback(goal)

    def _propose_fallback(self, goal: str) -> List[StateEdit]:
        """Fallback proposal logic without Bedrock."""
        edits: List[StateEdit] = []
        
        if "add" in goal.lower() or "create" in goal.lower():
            edits.append(StateEdit(
                operation="add_node",
                data={
                    "type": NodeType.ENTITY.value,
                    "label": goal.split()[-1] if goal.split() else "new_entity",
                    "properties": {}
                },
                reason=f"Goal requires: {goal}",
                priority=5
            ))

        return edits

    def get_last_usage(self) -> Optional[TokenUsage]:
        """Get token usage from last proposal."""
        return self.last_usage


class StateApplier:
    """Applies edits to the latent state."""

    @staticmethod
    def _coerce_node_type(value: Any) -> NodeType:
        """Return a valid NodeType, defaulting to ENTITY for unknown values."""
        if isinstance(value, NodeType):
            return value
        if isinstance(value, str):
            try:
                return NodeType(value)
            except ValueError:
                return NodeType.ENTITY
        return NodeType.ENTITY

    @staticmethod
    def apply_edits(state: LatentState, edits: List[StateEdit]) -> LatentState:
        """Apply a list of edits to the state."""
        new_state = LatentState(
            nodes=state.nodes.copy(),
            edges=state.edges.copy(),
            factors=state.factors.copy(),
            summaries=state.summaries.copy(),
            version=state.version
        )

        # Sort edits by priority (higher first)
        sorted_edits = sorted(edits, key=lambda e: e.priority, reverse=True)

        for edit in sorted_edits:
            StateApplier._apply_single_edit(new_state, edit)

        new_state.version += 1
        new_state.updated_at = state.updated_at
        return new_state

    @staticmethod
    def _apply_single_edit(state: LatentState, edit: StateEdit) -> None:
        """Apply a single edit to the state."""
        from datetime import datetime
        import hashlib

        operation = edit.operation
        data = edit.data

        if operation == "add_node":
            node_type = StateApplier._coerce_node_type(data.get("type"))
            label = data.get("label", "unnamed")
            node_id = data.get("id") or StateNormalizer._canonicalize_node_id(
                GraphNode(id="", type=node_type, label=label)
            )
            state.nodes[node_id] = GraphNode(
                id=node_id,
                type=node_type,
                label=label,
                properties=data.get("properties", {}),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

        elif operation == "remove_node":
            node_id = edit.target_id
            if node_id in state.nodes:
                del state.nodes[node_id]
                # Remove associated edges
                edges_to_remove = [
                    eid for eid, edge in state.edges.items()
                    if edge.source == node_id or edge.target == node_id
                ]
                for eid in edges_to_remove:
                    del state.edges[eid]

        elif operation == "add_edge":
            edge_id = data.get("id") or f"{data.get('source', '')}_{data.get('target', '')}_{data.get('type', 'depends_on')}"
            if not data.get("source") or not data.get("target"):
                return  # Skip invalid edge
            
            # Validate edge type
            edge_type_str = data.get("type", "depends_on")
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                # Fallback to depends_on if invalid type
                edge_type = EdgeType.DEPENDS_ON
            
            state.edges[edge_id] = GraphEdge(
                id=edge_id,
                source=data["source"],
                target=data["target"],
                type=edge_type,
                weight=data.get("weight", 1.0),
                properties=data.get("properties", {}),
                created_at=datetime.now()
            )

        elif operation == "remove_edge":
            edge_id = edit.target_id
            if edge_id in state.edges:
                del state.edges[edge_id]

        elif operation == "add_factor":
            factor_id = data.get("id") or f"factor_{len(state.factors)}"
            state.factors[factor_id] = Factor(
                id=factor_id,
                name=data["name"],
                nodes=set(data["nodes"]),
                constraint_type=data["constraint_type"],
                value=data.get("value"),
                properties=data.get("properties", {}),
                created_at=datetime.now()
            )

        elif operation == "remove_factor":
            factor_id = edit.target_id
            if factor_id in state.factors:
                del state.factors[factor_id]

        elif operation == "add_summary":
            summary_id = data.get("id") or f"summary_{len(state.summaries)}"
            summary_text = data.get("summary")
            scope = data.get("scope")
            if summary_text is None or scope is None:
                return
            summary_hash = hashlib.sha256(summary_text.encode()).hexdigest()[:16]
            state.summaries[summary_id] = MicroSummary(
                id=summary_id,
                scope=set(scope),
                summary=summary_text,
                hash=summary_hash,
                created_at=datetime.now()
            )

        elif operation == "update_node":
            node_id = edit.target_id
            if node_id in state.nodes:
                node = state.nodes[node_id]
                if "label" in data:
                    node.label = data["label"]
                if "properties" in data:
                    node.properties.update(data["properties"])
                node.updated_at = datetime.now()

        elif operation == "update_edge":
            edge_id = edit.target_id
            if edge_id in state.edges:
                edge = state.edges[edge_id]
                if "weight" in data:
                    edge.weight = data["weight"]
                if "properties" in data:
                    edge.properties.update(data["properties"])


class StateEditPipeline:
    """Complete pipeline: normalize → propose → apply."""

    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        self.normalizer = StateNormalizer()
        self.proposer = StateProposer(bedrock_client=bedrock_client)
        self.applier = StateApplier()

    def execute(
        self,
        current_state: LatentState,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> LatentState:
        """Execute the full pipeline."""
        # Normalize current state
        normalized_state = self.normalizer.normalize(current_state)

        # Propose edits
        edits = self.proposer.propose_edits(normalized_state, goal, context)

        # Apply edits
        new_state = self.applier.apply_edits(normalized_state, edits)

        return new_state

