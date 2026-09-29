"""graphrag/retrieval package.

Imports are kept lazy to avoid pulling in heavy ML deps (sentence_transformers,
torch) at collection time during testing.  Import classes directly when needed:

    from graphrag.retrieval.router import IntentRouter, QueryIntent
    from graphrag.retrieval.seed_selector import SeedSelector
"""

__all__ = ["IntentRouter", "QueryIntent", "SeedSelector"]


def __getattr__(name: str):
    if name in ("IntentRouter", "QueryIntent"):
        from .router import IntentRouter, QueryIntent  # noqa: F401
        return locals()[name]
    if name == "SeedSelector":
        from .seed_selector import SeedSelector
        return SeedSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
