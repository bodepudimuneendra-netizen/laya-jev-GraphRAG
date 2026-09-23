"""
graphrag/graph/neo4j_client.py

Neo4j 5.x client using the official Python driver over the Bolt protocol.

Key responsibilities
--------------------
- Node / edge CRUD (upsert semantics to avoid duplicates)
- Neighbour retrieval with PageRank scores (used by A* traversal)
- Batch PageRank computation via the Graph Data Science (GDS) library
- Vector-cosine index operations for seed-node selection

All queries go through the Bolt binary protocol; no HTTP is used at runtime,
which keeps latency at ~5 ms per hop (see benchmarks/hop_latency.py).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Session

from config.settings import settings
from .base import BaseGraphClient

logger = logging.getLogger(__name__)


class Neo4jClient(BaseGraphClient):
    """
    Thread-safe wrapper around the Neo4j Python driver.

    The driver maintains a connection pool internally; a single Neo4jClient
    instance is safe to share across threads.
    """

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j driver connected to %s", settings.neo4j_uri)

    def close(self) -> None:
        self._driver.close()

    # ── Context manager helpers ───────────────────────────────────────────────

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        with self._driver.session() as session:
            yield session

    # ── Schema setup ─────────────────────────────────────────────────────────

    def create_schema(self) -> None:
        """Create uniqueness constraints and vector index (run once)."""
        with self._session() as s:
            s.run(
                "CREATE CONSTRAINT unique_node_name IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE n.name IS UNIQUE"
            )
            # Vector index for cosine-similarity seed selection (Neo4j 5.11+)
            s.run(
                """
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (n:Entity) ON (n.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
            )
        logger.info("Neo4j schema created (constraints + vector index).")

    # ── Node operations ───────────────────────────────────────────────────────

    def upsert_node(
        self,
        name: str,
        label: str = "Entity",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Create or update a node.  The `name` property is the unique key.

        Parameters
        ----------
        name:
            Human-readable entity name (e.g., "Isaac Newton").
        label:
            Neo4j label (default: "Entity").
        properties:
            Additional properties to merge onto the node.
        """
        props = properties or {}
        props["name"] = name
        with self._session() as s:
            s.run(
                f"MERGE (n:{label} {{name: $name}}) SET n += $props",
                name=name,
                props=props,
            )

    def set_embedding(self, name: str, embedding: list[float]) -> None:
        """Store a pre-computed embedding vector on a node."""
        with self._session() as s:
            s.run(
                "MATCH (n:Entity {name: $name}) SET n.embedding = $emb",
                name=name,
                emb=embedding,
            )

    # ── Edge operations ───────────────────────────────────────────────────────

    def upsert_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Create or update a directed edge between two Entity nodes.

        Parameters
        ----------
        source / target:
            Node `name` values (must already exist).
        rel_type:
            Relationship type string (e.g., "INFLUENCES", "CAUSES").
        properties:
            Extra properties merged onto the relationship.
        """
        props = properties or {}
        query = (
            f"MATCH (a:Entity {{name: $src}}), (b:Entity {{name: $tgt}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props"
        )
        with self._session() as s:
            s.run(query, src=source, tgt=target, props=props)

    def delete_edge(self, source: str, target: str, rel_type: str) -> None:
        """Remove a specific directed edge (used by edge_verifier)."""
        query = (
            f"MATCH (a:Entity {{name: $src}})-[r:{rel_type}]->(b:Entity {{name: $tgt}}) "
            f"DELETE r"
        )
        with self._session() as s:
            s.run(query, src=source, tgt=target)

    # ── Neighbour retrieval (A* hot path) ─────────────────────────────────────

    def get_neighbors(self, node_name: str) -> list[dict[str, Any]]:
        """
        Return all outgoing edges from *node_name* with target info.

        Returns
        -------
        list of dicts with keys:
            target_name (str), type (str), pagerank (float)
        """
        query = """
        MATCH (n:Entity {name: $name})-[r]->(target:Entity)
        RETURN
            target.name  AS target_name,
            type(r)      AS type,
            coalesce(target.pagerank, 0.0) AS pagerank
        """
        with self._session() as s:
            result = s.run(query, name=node_name)
            return [record.data() for record in result]

    # ── Seed selection (vector cosine) ────────────────────────────────────────

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k nodes most similar to *query_embedding*.

        Uses the Neo4j vector index (requires Neo4j 5.11+ with GDS).

        Returns
        -------
        list of dicts with keys: name (str), score (float)
        """
        query = """
        CALL db.index.vector.queryNodes('entity_embedding', $k, $emb)
        YIELD node, score
        RETURN node.name AS name, score
        ORDER BY score DESC
        """
        with self._session() as s:
            result = s.run(query, k=top_k, emb=query_embedding)
            return [record.data() for record in result]

    # ── PageRank (GDS) ────────────────────────────────────────────────────────

    def run_pagerank(
        self,
        graph_name: str = "entity_graph",
        damping_factor: float = 0.85,
        max_iterations: int = 20,
    ) -> None:
        """
        Project the Entity graph into GDS memory and run PageRank.
        Results are written back as `n.pagerank` on every Entity node.
        """
        with self._session() as s:
            # Drop projection if it already exists
            s.run(
                "CALL gds.graph.exists($name) YIELD exists "
                "WITH exists WHERE exists "
                "CALL gds.graph.drop($name) YIELD graphName RETURN graphName",
                name=graph_name,
            )
            # Project
            s.run(
                """
                CALL gds.graph.project(
                    $name,
                    'Entity',
                    {*: {orientation: 'NATURAL'}}
                )
                """,
                name=graph_name,
            )
            # Run & write
            s.run(
                """
                CALL gds.pageRank.write($name, {
                    dampingFactor:  $df,
                    maxIterations:  $mi,
                    writeProperty:  'pagerank'
                })
                YIELD nodePropertiesWritten
                """,
                name=graph_name,
                df=damping_factor,
                mi=max_iterations,
            )
            # Clean up in-memory projection
            s.run("CALL gds.graph.drop($name)", name=graph_name)

        logger.info("PageRank computed and written to all Entity nodes.")

    # ── Community detection (Leiden → core-based fallback) ───────────────────

    def run_community_detection(self, graph_name: str = "entity_graph") -> None:
        """
        Run Leiden community detection via GDS and write communityId to nodes.
        Falls back to WCC (Weakly Connected Components) for sparse graphs.
        """
        with self._session() as s:
            # Re-project
            s.run(
                "CALL gds.graph.exists($name) YIELD exists WITH exists WHERE exists "
                "CALL gds.graph.drop($name) YIELD graphName RETURN graphName",
                name=graph_name,
            )
            s.run(
                "CALL gds.graph.project($name, 'Entity', {*: {orientation: 'UNDIRECTED'}})",
                name=graph_name,
            )
            try:
                s.run(
                    """
                    CALL gds.leiden.write($name, {
                        writeProperty: 'communityId'
                    })
                    YIELD communityCount
                    """,
                    name=graph_name,
                )
                logger.info("Leiden community detection complete.")
            except Exception:
                logger.warning("Leiden failed — falling back to WCC.")
                s.run(
                    """
                    CALL gds.wcc.write($name, {writeProperty: 'communityId'})
                    YIELD componentCount
                    """,
                    name=graph_name,
                )
            finally:
                s.run("CALL gds.graph.drop($name)", name=graph_name)
