"""graphrag/ingestion package.

Lazy imports to avoid pulling in heavy ML deps (sentence_transformers, torch)
at collection time during testing.

Import directly from submodules when needed:
    from graphrag.ingestion.chunker import NoulBoundaryChunker
    from graphrag.ingestion.entity_extractor import EntityExtractor
"""

__all__ = [
    "NoulBoundaryChunker",
    "EntityExtractor",
    "EdgeVerifier",
    "CommunityPartitioner",
    "OntologyAligner",
]


def __getattr__(name: str):
    if name == "NoulBoundaryChunker":
        from .chunker import NoulBoundaryChunker
        return NoulBoundaryChunker
    if name == "EntityExtractor":
        from .entity_extractor import EntityExtractor
        return EntityExtractor
    if name == "EdgeVerifier":
        from .edge_verifier import EdgeVerifier
        return EdgeVerifier
    if name == "CommunityPartitioner":
        from .community import CommunityPartitioner
        return CommunityPartitioner
    if name == "OntologyAligner":
        from .ontology_aligner import OntologyAligner
        return OntologyAligner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
