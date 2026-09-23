"""
graphrag/graph/factory.py

Factory to instantiate the configured Graph Database client.
"""

import logging
from config.settings import settings
from .base import BaseGraphClient

logger = logging.getLogger(__name__)


def get_graph_client() -> BaseGraphClient:
    """
    Instantiate and return the appropriate graph client based on GRAPH_DB_TYPE.
    """
    db_type = settings.graph_db_type.strip().lower()
    
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
        raise ValueError(f"Unknown GRAPH_DB_TYPE: {db_type}")
