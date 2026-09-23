"""
graphrag/retrieval/traversal/bfs.py

Score-Gated Breadth-First Search — single-hop traversal strategy.

From idea.md §3 (Traversal / Single-Hop):
    "Expand all neighbors, Laya filters out paths scoring < 0.6
     before final generation."

This module implements the BFS expansion layer.  After expanding all
direct neighbours of the seed node(s), each edge is scored by Laya and
paths below the configured threshold are pruned before being returned
to the pipeline for LLM synthesis.
"""

from __future__ import annotations

import logging
from typing import Any

from graphrag.graph.base import BaseGraphClient
from graphrag.graph.factory import get_graph_client
from graphrag.models.decision_factory import get_decision_model
from config.settings import settings

logger = logging.getLogger(__name__)


class ScoreGatedBFS:
    """
    Single-hop BFS retriever with Laya-based edge pruning.

    Expansion is kept to one hop for latency-sensitive Local queries
    (identified by the intent router).
    """

    def __init__(self, graph_client: BaseGraphClient | None = None) -> None:
        self._db    = graph_client or get_graph_client()
        self._laya  = get_decision_model()
        self._threshold = settings.bfs_prune_threshold  # default 0.6

    def retrieve(
        self,
        seed_nodes: list[str],
        user_query: str,
    ) -> list[dict[str, Any]]:
        """
        Expand all neighbours of *seed_nodes* and filter by Laya score.

        Parameters
        ----------
        seed_nodes:
            Starting entity names (output of seed_selector.py).
        user_query:
            Original user question used to score relevance.

        Returns
        -------
        list of dicts (sorted by score desc):
            source (str), target (str), edge_type (str), score (float)
        """
        results: list[dict[str, Any]] = []

        for seed in seed_nodes:
            edges = self._db.get_neighbors(seed)
            logger.debug("BFS: '%s' → %d neighbours", seed, len(edges))

            for edge in edges:
                context     = f"Node: {seed}. Edge: {edge['type']}. Target: {edge['target_name']}."
                instruction = f"Score relevance to: '{user_query}'"
                score = self._laya.score(context, instruction)

                if score >= self._threshold:
                    results.append({
                        "source":    seed,
                        "target":    edge["target_name"],
                        "edge_type": edge["type"],
                        "pagerank":  edge.get("pagerank", 0.0),
                        "score":     score,
                    })
                else:
                    logger.debug(
                        "BFS pruned: %s→%s (score=%.3f < %.2f)",
                        seed,
                        edge["target_name"],
                        score,
                        self._threshold,
                    )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
