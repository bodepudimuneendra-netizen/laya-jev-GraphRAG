"""
graphrag/graph/kuzu_client.py

Kùzu embedded graph database client.
Runs entirely locally in the Python process (no Docker required).
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import networkx as nx

from .base import BaseGraphClient

try:
    import kuzu
except ImportError:
    kuzu = None

logger = logging.getLogger(__name__)


class KuzuClient(BaseGraphClient):
    def __init__(self, db_path: str = "./kuzu_db") -> None:
        if kuzu is None:
            raise ImportError("Kuzu is not installed. Run `pip install kuzu`")
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        logger.info("Kùzu embedded database connected at %s", db_path)

        # In-memory vector store fallback since Kuzu's native vector index
        # is still maturing and we want guaranteed stability right now.
        self._embeddings: dict[str, np.ndarray] = {}

    def close(self) -> None:
        # Kuzu handles cleanup on process exit, but we can clear refs
        pass

    def create_schema(self) -> None:
        try:
            self.conn.execute("CREATE NODE TABLE Entity(name STRING, pagerank DOUBLE, communityId INT64, PRIMARY KEY (name))")
            self.conn.execute("CREATE REL TABLE RELATES_TO(FROM Entity TO Entity, type STRING)")
            logger.info("Kùzu schema created.")
        except RuntimeError as e:
            if "already exists" in str(e).lower():
                logger.info("Kùzu schema already exists.")
            else:
                raise

    def upsert_node(self, name: str, label: str = "Entity", properties: dict[str, Any] | None = None) -> None:
        # Kuzu's MERGE syntax
        query = "MERGE (n:Entity {name: $name})"
        self.conn.execute(query, parameters={"name": name})

    def set_embedding(self, name: str, embedding: list[float]) -> None:
        self._embeddings[name] = np.array(embedding, dtype=np.float32)

    def upsert_edge(self, source: str, target: str, rel_type: str, properties: dict[str, Any] | None = None) -> None:
        query = """
        MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
        MERGE (a)-[r:RELATES_TO {type: $rel}]->(b)
        """
        self.conn.execute(query, parameters={"src": source, "tgt": target, "rel": rel_type})

    def delete_edge(self, source: str, target: str, rel_type: str) -> None:
        query = """
        MATCH (a:Entity {name: $src})-[r:RELATES_TO {type: $rel}]->(b:Entity {name: $tgt})
        DELETE r
        """
        self.conn.execute(query, parameters={"src": source, "tgt": target, "rel": rel_type})

    def get_neighbors(self, node_name: str) -> list[dict[str, Any]]:
        query = """
        MATCH (n:Entity {name: $name})-[r:RELATES_TO]->(target:Entity)
        RETURN target.name, r.type, target.pagerank
        """
        results = self.conn.execute(query, parameters={"name": node_name})
        
        neighbors = []
        while results.has_next():
            row = results.get_next()
            # Kuzu returns a list for the row values
            neighbors.append({
                "target_name": row[0],
                "type":        row[1],
                "pagerank":    row[2] if row[2] is not None else 0.0,
            })
        return neighbors

    def vector_search(self, query_embedding: list[float], top_k: int = 20) -> list[dict[str, Any]]:
        if not self._embeddings:
            return []
        q_vec = np.array(query_embedding, dtype=np.float32)
        names = list(self._embeddings.keys())
        matrix = np.stack(list(self._embeddings.values()))
        sims = (matrix @ q_vec) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec) + 1e-9)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [{"name": names[i], "score": float(sims[i])} for i in top_indices]

    def _get_networkx_graph(self) -> nx.DiGraph:
        query = "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.name, b.name"
        results = self.conn.execute(query)
        G = nx.DiGraph()
        while results.has_next():
            row = results.get_next()
            G.add_edge(row[0], row[1])
        return G

    def run_pagerank(self, graph_name: str = "entity_graph", damping_factor: float = 0.85, max_iterations: int = 20) -> None:
        G = self._get_networkx_graph()
        pr = nx.pagerank(G, alpha=damping_factor, max_iter=max_iterations)
        
        for node, score in pr.items():
            query = "MATCH (n:Entity {name: $name}) SET n.pagerank = $score"
            self.conn.execute(query, parameters={"name": node, "score": score})
        logger.info("Kùzu PageRank computed via NetworkX and written back.")

    def run_community_detection(self, graph_name: str = "entity_graph") -> None:
        G = self._get_networkx_graph().to_undirected()
        components = nx.connected_components(G)
        
        for c_id, comp in enumerate(components):
            for node in comp:
                query = "MATCH (n:Entity {name: $name}) SET n.communityId = $cid"
                self.conn.execute(query, parameters={"name": node, "cid": c_id})
        logger.info("Kùzu WCC computed via NetworkX and written back.")
