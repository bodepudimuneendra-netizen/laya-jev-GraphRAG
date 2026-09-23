"""
graphrag/graph/age_client.py

Apache AGE (Postgres) client implementing the full BaseGraphClient interface.

Since standard Apache AGE does not include pgvector or GDS algorithms out of
the box, this client handles vector search via a lightweight in-memory NumPy index,
and computes PageRank/Communities by pulling the graph into NetworkX locally.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator
import numpy as np
import psycopg2
import psycopg2.extras
import networkx as nx

from config.settings import settings
from .base import BaseGraphClient

logger = logging.getLogger(__name__)


class AGEClient(BaseGraphClient):
    def __init__(self) -> None:
        self._conn = psycopg2.connect(
            host=settings.age_host,
            port=settings.age_port,
            user=settings.age_user,
            password=settings.age_password,
            dbname=settings.age_database,
        )
        self._conn.autocommit = True
        self._setup_extension()
        
        # In-memory vector store fallback
        self._embeddings: dict[str, np.ndarray] = {}
        
        logger.info("Apache AGE connected at %s:%s", settings.age_host, settings.age_port)

    def _setup_extension(self) -> None:
        with self._cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")

    @contextmanager
    def _cursor(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur

    def close(self) -> None:
        self._conn.close()

    def create_schema(self) -> None:
        """Initialize the AGE graph if it doesn't exist."""
        with self._cursor() as cur:
            cur.execute("SELECT create_graph('entity_graph');")
        logger.info("Apache AGE graph 'entity_graph' initialized.")

    def upsert_node(self, name: str, label: str = "Entity", properties: dict[str, Any] | None = None) -> None:
        # Note: Apache AGE uses standard cypher string interpolation, which must be sanitized in production.
        cypher = f"MERGE (n:{label} {{name: '{name}'}})"
        with self._cursor() as cur:
            cur.execute(f"SELECT * FROM cypher('entity_graph', $$ {cypher} $$) AS (v agtype);")

    def set_embedding(self, name: str, embedding: list[float]) -> None:
        """Store embedding in memory since pgvector is not assumed present."""
        self._embeddings[name] = np.array(embedding, dtype=np.float32)

    def upsert_edge(self, source: str, target: str, rel_type: str, properties: dict[str, Any] | None = None) -> None:
        cypher = (
            f"MATCH (a:Entity {{name: '{source}'}}), (b:Entity {{name: '{target}'}}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )
        with self._cursor() as cur:
            cur.execute(f"SELECT * FROM cypher('entity_graph', $$ {cypher} $$) AS (v agtype);")

    def delete_edge(self, source: str, target: str, rel_type: str) -> None:
        cypher = (
            f"MATCH (a:Entity {{name: '{source}'}})-[r:{rel_type}]->(b:Entity {{name: '{target}'}}) "
            f"DELETE r"
        )
        with self._cursor() as cur:
            cur.execute(f"SELECT * FROM cypher('entity_graph', $$ {cypher} $$) AS (v agtype);")

    def get_neighbors(self, node_name: str) -> list[dict[str, Any]]:
        cypher = (
            f"MATCH (n:Entity {{name: '{node_name}'}})-[r]->(target:Entity) "
            f"RETURN target.name AS target_name, type(r) AS type, "
            f"coalesce(target.pagerank, 0.0) AS pagerank"
        )
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM cypher('entity_graph', $$ {cypher} $$) "
                f"AS (target_name agtype, type agtype, pagerank agtype);"
            )
            rows = cur.fetchall()

        return [
            {
                "target_name": str(r["target_name"]).strip('"'),
                "type":        str(r["type"]).strip('"'),
                "pagerank":    float(str(r["pagerank"])),
            }
            for r in rows
        ]

    def vector_search(self, query_embedding: list[float], top_k: int = 20) -> list[dict[str, Any]]:
        """In-memory cosine similarity search."""
        if not self._embeddings:
            return []
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        names = list(self._embeddings.keys())
        matrix = np.stack(list(self._embeddings.values()))
        
        # Compute cosine similarity
        sims = (matrix @ q_vec) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec) + 1e-9)
        
        # Get top k indices
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [{"name": names[i], "score": float(sims[i])} for i in top_indices]

    def _get_networkx_graph(self) -> nx.DiGraph:
        """Extract graph to NetworkX for Python-side algorithms."""
        cypher = "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS src, b.name AS tgt"
        with self._cursor() as cur:
            cur.execute(f"SELECT * FROM cypher('entity_graph', $$ {cypher} $$) AS (src agtype, tgt agtype);")
            rows = cur.fetchall()
            
        G = nx.DiGraph()
        for r in rows:
            G.add_edge(str(r["src"]).strip('"'), str(r["tgt"]).strip('"'))
        return G

    def run_pagerank(self, graph_name: str = "entity_graph", damping_factor: float = 0.85, max_iterations: int = 20) -> None:
        G = self._get_networkx_graph()
        pr = nx.pagerank(G, alpha=damping_factor, max_iter=max_iterations)
        
        with self._cursor() as cur:
            for node, score in pr.items():
                cypher = f"MATCH (n:Entity {{name: '{node}'}}) SET n.pagerank = {score}"
                cur.execute(f"SELECT * FROM cypher('entity_graph', $$ {cypher} $$) AS (v agtype);")
        logger.info("Apache AGE PageRank computed via NetworkX and written back.")

    def run_community_detection(self, graph_name: str = "entity_graph") -> None:
        G = self._get_networkx_graph().to_undirected()
        components = nx.connected_components(G)
        
        with self._cursor() as cur:
            for c_id, comp in enumerate(components):
                for node in comp:
                    cypher = f"MATCH (n:Entity {{name: '{node}'}}) SET n.communityId = {c_id}"
                    cur.execute(f"SELECT * FROM cypher('entity_graph', $$ {cypher} $$) AS (v agtype);")
        logger.info("Apache AGE WCC computed via NetworkX and written back.")
