"""
graphrag/graph/factory.py

Factory to instantiate the configured Graph Database client.

Reads GRAPH_DB_BACKEND (canonical) or GRAPH_DB_TYPE (legacy alias).
Accepted values: neo4j | memgraph | age | kuzu
'age' is normalised to 'postgres_age' internally.
"""

import logging
from config.settings import settings
from .base import BaseGraphClient

logger = logging.getLogger(__name__)


def _resolve_backend() -> str:
    """
    Resolve the graph DB backend from settings.

    Priority:
      1. GRAPH_DB_BACKEND (canonical, as shown in README)
      2. GRAPH_DB_TYPE    (legacy alias)
    Normalises 'age' -> 'postgres_age'.
    """
    backend = (settings.graph_db_backend or settings.graph_db_type or "neo4j").strip().lower()
    # Normalise the short alias used in README
    if backend == "age":
        backend = "postgres_age"
    return backend


def get_graph_client() -> BaseGraphClient:
    """
    Instantiate and return the appropriate graph client based on GRAPH_DB_BACKEND.

    Supported backends
    ------------------
    neo4j        -> Neo4jClient       (Bolt + GDS, production default)
    memgraph     -> MemgraphClient    (Bolt + MAGE, in-memory analytics)
    age          -> AGEClient         (Apache AGE / PostgreSQL + Cypher)
    kuzu         -> KuzuClient        (embedded, no Docker, local dev)
    """
    db_type = _resolve_backend()

    if db_type == "neo4j":
        from .neo4j_client import Neo4jClient
        logger.info("Initializing Neo4j Graph Client")
        return Neo4jClient()

    elif db_type == "postgres_age":
        from .age_client import AGEClient
        logger.info("Initializing Apache AGE (Postgres) Graph Client")
        return AGEClient()

    elif db_type == "memgraph":
        from .memgraph_client import MemgraphClient
        logger.info("Initializing Memgraph Graph Client")
        return MemgraphClient()

    elif db_type == "kuzu":
        from .kuzu_client import KuzuClient
        logger.info("Initializing Kuzu Embedded Graph Client")
        return KuzuClient()

    else:
        raise ValueError(
            f"Unknown GRAPH_DB_BACKEND: '{db_type}'. "
            f"Choose one of: neo4j, memgraph, age, kuzu"
        )
