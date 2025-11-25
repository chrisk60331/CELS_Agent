"""Core latent state structures for compressed reasoning."""
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class NodeType(str, Enum):
    """Types of nodes in the typed graph."""
    ENTITY = "entity"
    ACTION = "action"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    RESOURCE = "resource"
    STATE = "state"
    FILE = "file"


class EdgeType(str, Enum):
    """Types of edges in the typed graph."""
    DEPENDS_ON = "depends_on"
    PRODUCES = "produces"
    REQUIRES = "requires"
    CONSTRAINS = "constrains"
    PRECEDES = "precedes"
    CONTAINS = "contains"


class GraphNode(BaseModel):
    """A node in the typed graph."""
    id: str = Field(..., description="Unique identifier")
    type: NodeType = Field(..., description="Type of node")
    label: str = Field(..., description="Human-readable label")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class GraphEdge(BaseModel):
    """An edge in the typed graph."""
    id: str = Field(..., description="Unique identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: EdgeType = Field(..., description="Type of edge")
    weight: float = Field(default=1.0, description="Edge weight/strength")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    created_at: datetime = Field(default_factory=datetime.now)


class Factor(BaseModel):
    """A factor representing a constraint or relationship."""
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Factor name")
    nodes: Set[str] = Field(..., description="Node IDs involved in this factor")
    constraint_type: str = Field(..., description="Type of constraint")
    value: Any = Field(..., description="Factor value or constraint")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    created_at: datetime = Field(default_factory=datetime.now)


class MicroSummary(BaseModel):
    """A canonical micro-summary for compression."""
    id: str = Field(..., description="Unique identifier")
    scope: Set[str] = Field(..., description="Node IDs this summary covers")
    summary: str = Field(..., description="Compressed summary text")
    hash: str = Field(..., description="Content hash for deduplication")
    created_at: datetime = Field(default_factory=datetime.now)


class LatentState(BaseModel):
    """The complete compressed latent state."""
    nodes: Dict[str, GraphNode] = Field(default_factory=dict, description="All graph nodes")
    edges: Dict[str, GraphEdge] = Field(default_factory=dict, description="All graph edges")
    factors: Dict[str, Factor] = Field(default_factory=dict, description="All factors")
    summaries: Dict[str, MicroSummary] = Field(default_factory=dict, description="All micro-summaries")
    version: int = Field(default=0, description="State version for tracking changes")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all edges from a node."""
        return [e for e in self.edges.values() if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all edges to a node."""
        return [e for e in self.edges.values() if e.target == node_id]

    def get_factors_for_node(self, node_id: str) -> List[Factor]:
        """Get all factors involving a node."""
        return [f for f in self.factors.values() if node_id in f.nodes]

    def get_summaries_for_scope(self, node_ids: Set[str]) -> List[MicroSummary]:
        """Get summaries that cover any of the given nodes."""
        return [s for s in self.summaries.values() if s.scope & node_ids]

