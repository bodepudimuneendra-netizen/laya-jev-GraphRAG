"""graphrag/graph package.

Imports are kept lazy to avoid pulling in optional DB drivers (neo4j, psycopg2,
kuzu) that may not be installed in a test environment.

Import directly from submodules when needed:
    from graphrag.graph.neo4j_client import Neo4jClient
    from graphrag.graph.factory import get_graph_client
"""

__all__ = ["BaseGraphClient", "Neo4jClient", "MemgraphClient", "AGEClient", "KuzuClient", "get_graph_client"]


def __getattr__(name: str):
    if name == "BaseGraphClient":
        from .base import BaseGraphClient
        return BaseGraphClient
    if name == "Neo4jClient":
        from .neo4j_client import Neo4jClient
        return Neo4jClient
    if name == "MemgraphClient":
        from .memgraph_client import MemgraphClient
        return MemgraphClient
    if name == "AGEClient":
        from .age_client import AGEClient
        return AGEClient
    if name == "KuzuClient":
        from .kuzu_client import KuzuClient
        return KuzuClient
    if name == "get_graph_client":
        from .factory import get_graph_client
        return get_graph_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
