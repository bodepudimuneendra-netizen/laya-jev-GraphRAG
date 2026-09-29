"""graphrag/retrieval/traversal package.

Lazy imports to avoid pulling in heavy ML/DB deps at collection time.

    from graphrag.retrieval.traversal.astar import LayaGraphNavigator
    from graphrag.retrieval.traversal.bfs import ScoreGatedBFS
"""

__all__ = ["LayaGraphNavigator", "ScoreGatedBFS"]


def __getattr__(name: str):
    if name == "LayaGraphNavigator":
        from .astar import LayaGraphNavigator
        return LayaGraphNavigator
    if name == "ScoreGatedBFS":
        from .bfs import ScoreGatedBFS
        return ScoreGatedBFS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
