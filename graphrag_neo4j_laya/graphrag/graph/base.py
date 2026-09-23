"""
graphrag/graph/base.py

Abstract Base Class defining the universal interface for all Graph Databases
used in this GraphRAG pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGraphClient(ABC):
    """
    Standardised interface for GraphRAG graph operations.
    """

    @abstractmethod
    def close(self) -> None:
        """Close any active connections to the database."""
        pass

    # ── Schema setup ─────────────────────────────────────────────────────────

    @abstractmethod
    def create_schema(self) -> None:
        """Create uniqueness constraints and vector indexes (run once)."""
        pass

    # ── Node operations ───────────────────────────────────────────────────────

    @abstractmethod
    def upsert_node(
        self,
        name: str,
        label: str = "Entity",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a node. The `name` property is the unique key."""
        pass

    @abstractmethod
    def set_embedding(self, name: str, embedding: list[float]) -> None:
        """Store a pre-computed embedding vector on a node."""
        pass

    # ── Edge operations ───────────────────────────────────────────────────────

    @abstractmethod
    def upsert_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a directed edge between two Entity nodes."""
        pass

    @abstractmethod
    def delete_edge(self, source: str, target: str, rel_type: str) -> None:
        """Remove a specific directed edge."""
        pass

    # ── Retrieval (A* hot path) ───────────────────────────────────────────────

    @abstractmethod
    def get_neighbors(self, node_name: str) -> list[dict[str, Any]]:
        """
        Return all outgoing edges from *node_name*.
        
        Returns:
            list of dicts with keys: target_name (str), type (str), pagerank (float)
        """
        pass

    # ── Seed selection (Vector search) ────────────────────────────────────────

    @abstractmethod
    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k nodes most similar to *query_embedding*.
        
        Returns:
            list of dicts with keys: name (str), score (float)
        """
        pass

    # ── Global Algorithms ─────────────────────────────────────────────────────

    @abstractmethod
    def run_pagerank(
        self,
        graph_name: str = "entity_graph",
        damping_factor: float = 0.85,
        max_iterations: int = 20,
    ) -> None:
        """Compute PageRank and write the score back to all Entity nodes."""
        pass

    @abstractmethod
    def run_community_detection(self, graph_name: str = "entity_graph") -> None:
        """Compute Community IDs (e.g. Leiden/WCC) and write them to nodes."""
        pass
