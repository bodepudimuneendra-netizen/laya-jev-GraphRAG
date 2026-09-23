"""
graphrag/ingestion/edge_verifier.py

Continuous Edge Pruning via Laya.

From idea.md §3 (Ingestion / Edge Verification):
    "Continuous Edge Pruning: Run Laya in the background, asking for a score
     on edge validity.  Delete mathematically illogical connections."

The verifier iterates over all edges in Neo4j and asks Laya to evaluate
whether each relationship is logically valid.  Edges scoring below the
prune threshold are deleted from the graph.

Usage
-----
    verifier = EdgeVerifier(neo4j_client)
    stats = verifier.run_full_verification(prune_threshold=0.3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from graphrag.graph.base import BaseGraphClient
from graphrag.graph.factory import get_graph_client
from graphrag.models.decision_factory import get_decision_model
from config.settings import settings

logger = logging.getLogger(__name__)

_ALL_EDGES_QUERY = """
MATCH (a:Entity)-[r]->(b:Entity)
RETURN a.name AS source, b.name AS target, type(r) AS rel_type
LIMIT $limit
"""


@dataclass
class VerificationStats:
    total_checked: int = 0
    pruned:        int = 0
    kept:          int = 0
    pruned_edges:  list[tuple[str, str, str]] = field(default_factory=list)


class EdgeVerifier:
    """
    Background Laya-based edge pruner.

    For each edge, Laya is asked: "Is the relationship between A and B
    via [REL_TYPE] logically valid?"  Low-scoring edges are removed.
    """

    def __init__(self, graph_client: BaseGraphClient | None = None) -> None:
        self._db    = graph_client or get_graph_client()
        self._laya  = get_decision_model()

    def _score_edge(self, source: str, rel_type: str, target: str) -> float:
        context = f"Source entity: {source}. Relationship type: {rel_type}. Target entity: {target}."
        instruction = "Is this a logically valid and meaningful relationship between these two entities?"
        return self._laya.score(context, instruction)

    def _fetch_all_edges(self, limit: int = 10_000) -> list[dict[str, Any]]:
        # This is a bit tricky: Kuzu uses self.conn, Neo4j/AGE/Memgraph use _driver or _cursor
        # For full abstraction, we should probably add `execute_query` to BaseGraphClient.
        # For now, let's just use the fact that this is specific to Neo4j/AGE/Memgraph's Cypher.
        # Actually, let's implement a quick workaround for all clients or add an abstract method.
        # I'll just use a try-except block here for now to handle the different underlying drivers.
        
        edges = []
        try:
            # Neo4j / Memgraph
            with self._db._session() as s:
                result = s.run(_ALL_EDGES_QUERY, limit=limit)
                edges = [record.data() for record in result]
        except AttributeError:
            try:
                # AGE (Postgres)
                with self._db._cursor() as cur:
                    query = _ALL_EDGES_QUERY.replace("$limit", str(limit))
                    cur.execute(f"SELECT * FROM cypher('entity_graph', $$ {query} $$) AS (source agtype, target agtype, rel_type agtype);")
                    rows = cur.fetchall()
                    edges = [{"source": str(r["source"]).strip('"'), "target": str(r["target"]).strip('"'), "rel_type": str(r["rel_type"]).strip('"')} for r in rows]
            except AttributeError:
                # Kuzu
                query = _ALL_EDGES_QUERY.replace("$limit", "$limit") # Kuzu supports parameters
                results = self._db.conn.execute(query, parameters={"limit": limit})
                while results.has_next():
                    row = results.get_next()
                    edges.append({"source": row[0], "target": row[1], "rel_type": row[2]})
                    
        return edges

    def run_full_verification(
        self,
        prune_threshold: float = 0.3,
        batch_size: int = 50,
        limit: int = 10_000,
    ) -> VerificationStats:
        """
        Scan and prune all edges in the graph.

        Parameters
        ----------
        prune_threshold:
            Edges scoring below this are deleted.  Lower = more aggressive.
        batch_size:
            Number of edges to evaluate per Laya batch.
        limit:
            Maximum edges to check in one run.

        Returns
        -------
        VerificationStats
        """
        edges = self._fetch_all_edges(limit=limit)
        stats = VerificationStats(total_checked=len(edges))
        logger.info("Edge verification: checking %d edges (threshold=%.2f)", len(edges), prune_threshold)

        for edge in edges:
            src, tgt, rel = edge["source"], edge["target"], edge["rel_type"]
            score = self._score_edge(src, rel, tgt)

            if score < prune_threshold:
                self._db.delete_edge(src, tgt, rel)
                stats.pruned += 1
                stats.pruned_edges.append((src, tgt, rel))
                logger.debug("Pruned edge: %s -[%s]-> %s (score=%.3f)", src, rel, tgt, score)
            else:
                stats.kept += 1

        logger.info(
            "Edge verification complete: %d checked, %d pruned, %d kept.",
            stats.total_checked,
            stats.pruned,
            stats.kept,
        )
        return stats
