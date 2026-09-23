"""
graphrag/graph/memgraph_client.py

Memgraph client using the Neo4j Python driver (Bolt protocol).

Memgraph is highly compatible with Neo4j. The primary differences are:
1. Memgraph is in-memory (much faster for traversal).
2. It uses MAGE (Memgraph Advanced Graph Extensions) instead of GDS for algorithms.
"""

from __future__ import annotations
import logging
from typing import Any
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class MemgraphClient(Neo4jClient):
    """
    Client for Memgraph. Inherits from Neo4jClient since both use Cypher and Bolt.
    Overrides schema creation and global algorithms to use Memgraph's MAGE.
    """

    def create_schema(self) -> None:
        """Create uniqueness constraints for Memgraph."""
        with self._session() as s:
            s.run("CREATE CONSTRAINT ON (n:Entity) ASSERT n.name IS UNIQUE")
            # Note: Memgraph supports vector indexes natively, but the syntax differs.
            # For simplicity, we create the standard constraint here.
        logger.info("Memgraph schema created (constraints).")

    def run_pagerank(
        self,
        graph_name: str = "entity_graph",
        damping_factor: float = 0.85,
        max_iterations: int = 20,
    ) -> None:
        """
        Compute PageRank using Memgraph's MAGE (pagerank module).
        """
        query = """
        CALL pagerank.get()
        YIELD node, rank
        SET node.pagerank = rank
        """
        with self._session() as s:
            s.run(query)
        logger.info("Memgraph PageRank computed and written.")

    def run_community_detection(self, graph_name: str = "entity_graph") -> None:
        """
        Compute Community IDs using Memgraph's MAGE (community_detection).
        """
        query = """
        CALL community_detection.get()
        YIELD node, community_id
        SET node.communityId = community_id
        """
        with self._session() as s:
            s.run(query)
        logger.info("Memgraph Community Detection complete.")
