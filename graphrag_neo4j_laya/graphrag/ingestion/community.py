"""
graphrag/ingestion/community.py

Core-based Hierarchical Community Detection.

From idea.md §3 (Ingestion / Community Partitioning):
    "Core-based Hierarchical: Replace standard Leiden with core-based
     clustering for sparse graph stability."

This module wraps Neo4j GDS community detection (Leiden → WCC fallback)
and additionally provides a pure-Python NetworkX-based k-core decomposition
for offline analysis and debugging.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from graphrag.graph.base import BaseGraphClient
from graphrag.graph.factory import get_graph_client

logger = logging.getLogger(__name__)


class CommunityPartitioner:
    """
    Runs community detection via the Graph Database abstraction.
    """

    def __init__(self, graph_client: BaseGraphClient | None = None) -> None:
        self._db = graph_client or get_graph_client()

    def partition(self, graph_name: str = "entity_graph") -> None:
        """
        Run community detection and write communityId to all nodes.
        Delegates to BaseGraphClient which handles Leiden / WCC.
        """
        self._db.run_community_detection(graph_name)

    # ── Offline NetworkX k-core decomposition ─────────────────────────────────

    @staticmethod
    def kcore_decomposition(
        edges: list[tuple[str, str]],
        k: int = 2,
    ) -> dict[str, int]:
        """
        Compute k-core decomposition of a graph for offline analysis.

        Parameters
        ----------
        edges:
            List of (source, target) name pairs.
        k:
            Minimum degree for core membership.

        Returns
        -------
        dict
            {node_name: core_number} for all nodes.
        """
        G = nx.DiGraph()
        G.add_edges_from(edges)
        G_undirected = G.to_undirected()
        core_numbers = nx.core_number(G_undirected)
        logger.info(
            "K-core decomposition: %d nodes, max core = %d",
            len(core_numbers),
            max(core_numbers.values(), default=0),
        )
        return core_numbers

    def fetch_edge_list(self) -> list[tuple[str, str]]:
        """Fetch all edges from Graph DB as (source, target) name pairs."""
        # Simple hack similarly to edge_verifier to abstract over driver differences
        edges = []
        try:
            with self._db._session() as s:
                result = s.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS src, b.name AS tgt")
                edges = [(r["src"], r["tgt"]) for r in result]
        except AttributeError:
            try:
                with self._db._cursor() as cur:
                    cur.execute("SELECT * FROM cypher('entity_graph', $$ MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS src, b.name AS tgt $$) AS (src agtype, tgt agtype);")
                    rows = cur.fetchall()
                    edges = [(str(r["src"]).strip('"'), str(r["tgt"]).strip('"')) for r in rows]
            except AttributeError:
                results = self._db.conn.execute("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.name, b.name")
                while results.has_next():
                    row = results.get_next()
                    edges.append((row[0], row[1]))
        return edges
