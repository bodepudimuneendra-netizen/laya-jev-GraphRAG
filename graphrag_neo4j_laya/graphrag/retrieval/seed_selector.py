"""
graphrag/retrieval/seed_selector.py

Two-stage seed node selection:

  Stage 1 — sentence-transformers cosine similarity → top-N candidates
  Stage 2 — RAGatouille (ColBERT late-interaction) re-ranks → top-K seeds

From idea.md §3 (Pre-Retrieval / Seed Selection):
    "ColBERT late-interaction pulls top 20 nodes; Laya score acts as a
     cross-encoder to isolate the top 2 seeds."

Implementation note
-------------------
RAGatouille wraps ColBERT behind a simple API.  We use it in "retrieve"
mode on a small in-memory index built from the candidate node descriptions.
For production-scale graphs, persist the ColBERT index to disk.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from graphrag.graph.base import BaseGraphClient
from graphrag.graph.factory import get_graph_client
from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embed_model_id)
    return SentenceTransformer(settings.embed_model_id)


class SeedSelector:
    """
    Identifies the best seed nodes in the graph for a given user query.

    Pipeline
    --------
    1. Embed the query with sentence-transformers.
    2. Use Graph DB's vector index to retrieve top-N candidate nodes.
    3. Re-rank with RAGatouille (ColBERT) to get the final top-K seeds.
    """

    def __init__(self, graph_client: BaseGraphClient | None = None) -> None:
        self._db     = graph_client or get_graph_client()
        self._embedder = _get_embedder()
        self._top_n  = settings.seed_colbert_top_n   # 20
        self._top_k  = settings.seed_final_top_k      # 2

    # ── Stage 1: dense retrieval ──────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        vec = self._embedder.encode(query, normalize_embeddings=True)
        return vec.tolist()

    def _dense_candidates(self, query: str) -> list[dict[str, Any]]:
        """Return top-N nodes from the Graph DB vector index."""
        query_vec = self._embed_query(query)
        candidates = self._db.vector_search(query_vec, top_k=self._top_n)
        logger.debug("Dense retrieval: %d candidates", len(candidates))
        return candidates

    # ── Stage 2: Semantic Validation (Score primitive) ────────────────────────
    
    def _semantic_validate(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Re-rank *candidates* using the decision model Score primitive.
        
        From functions.md: "Score each candidate node against the user query.
        Filter out irrelevant entry points."
        """
        if not candidates:
            return []
            
        from graphrag.models.decision_factory import get_decision_model  # noqa: PLC0415
        model = get_decision_model()
        
        scored: list[tuple[float, dict]] = []
        instruction = f"Score the direct relevance of this entity as a starting point to answer: '{query}'"
        
        for cand in candidates:
            context = f"Entity: {cand.get('name', '')}. Description: {cand.get('text', cand.get('name', ''))}"
            try:
                s = model.score(context, instruction)
                scored.append((s, cand))
            except Exception as e:
                logger.warning("Score evaluation failed for %s: %s", cand.get("name"), e)
                scored.append((0.0, cand))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        best_seeds = [c for s, c in scored[:self._top_k]]
        
        logger.info(
            "Semantic validation → top seeds: %s",
            [(s, c.get("name")) for s, c in scored[:self._top_k]]
        )
        return best_seeds

    # ── Public API ────────────────────────────────────────────────────────────

    def select(self, user_query: str) -> list[str]:
        """
        Return the names of the top-K seed nodes for *user_query*.

        Parameters
        ----------
        user_query:
            The user's natural-language question.

        Returns
        -------
        list[str]
            Ordered list of node names (best seed first).
        """
        candidates = self._dense_candidates(user_query)
        if not candidates:
            logger.warning("No candidate nodes found — check Neo4j vector index.")
            return []

        seeds = self._semantic_validate(user_query, candidates)
        seed_names = [s["name"] for s in seeds]
        logger.info("Final seeds: %s", seed_names)
        return seed_names
