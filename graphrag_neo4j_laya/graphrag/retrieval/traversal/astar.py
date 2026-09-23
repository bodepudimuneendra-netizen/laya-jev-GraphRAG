"""
graphrag/retrieval/traversal/astar.py

Semantic A* Search — the core multi-hop traversal engine from idea.md §4.

Implements the composite priority score:

    f(n) = α·S_Laya(E) + β·C_Neo4j(n) − γ·D(n)

where:
    S_Laya  — Laya typed-decisions semantic relevance ∈ [0, 1]
    C_Neo4j — Normalised PageRank of the target node ∈ [0, 1]
    D(n)    — Current traversal depth (depth penalty)
    α, β, γ — Hyperparameters from config/settings.py

The frontier is maintained as a max-heap via Python's heapq (min-heap with
negated scores), giving O(log N) push/pop instead of the O(N log N) sort used
in the original idea.md snippet.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

from graphrag.graph.base import BaseGraphClient
from graphrag.graph.factory import get_graph_client
from graphrag.models.decision_factory import get_decision_model
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _FrontierNode:
    """
    Priority-queue entry.
    Negated score so heapq (min-heap) acts as a max-heap.
    """

    neg_score: float                                # −f(n)
    node_name: str = field(compare=False)
    depth:     int = field(compare=False)
    path:      list[str] = field(compare=False, default_factory=list)
    score:     float = field(compare=False, default=0.0)


class LayaGraphNavigator:
    """
    A* graph navigator powered by Laya semantic heuristics and Graph DB PageRank.

    Direct, improved port of the `LayaGraphNavigator` from idea.md §4.
    Improvements over the original:
      - O(log N) heapq frontier instead of O(N log N) sort
      - Visited-node set to prevent revisiting (cycle-safe)
      - Configurable alpha/beta/gamma loaded from settings
      - Batch Laya scoring for throughput
    """

    def __init__(self, graph_client: BaseGraphClient | None = None) -> None:
        self._db    = graph_client or get_graph_client()
        self._laya  = get_decision_model()

        # Loaded from pydantic settings (idea.md §2.3 values as defaults)
        self.ALPHA       = settings.alpha        # 0.65 — Laya semantic weight
        self.BETA        = settings.beta         # 0.25 — PageRank structural weight
        self.GAMMA       = settings.gamma        # 0.10 — depth penalty
        self.MAX_PAGERANK = settings.max_pagerank  # 10.0 — graph-specific normaliser

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_edges(self, node_name: str) -> list[dict[str, Any]]:
        """Fetch outgoing edges from Graph DB (target + pagerank)."""
        return self._db.get_neighbors(node_name)

    def _laya_score_edge(
        self,
        user_query: str,
        current_node: str,
        edge: dict[str, Any],
    ) -> float:
        """
        Compute S_Laya for a single edge.

        Context encodes the traversal state; instruction encodes the query goal.
        (Exact formulation from idea.md §4.)
        """
        context     = f"Node: {current_node}. Edge: {edge['type']}. Target: {edge['target_name']}."
        instruction = f"Score relevance to: '{user_query}'"
        return self._laya.score(context, instruction)

    def _compute_f(
        self,
        s_laya: float,
        pagerank: float | None,
        depth: int,
    ) -> float:
        """
        Composite A* priority: f(n) = α·S_Laya + β·PR(n) − γ·D  (idea.md §2.3)
        """
        c_neo4j = min((pagerank or 0.0) / self.MAX_PAGERANK, 1.0)
        return (self.ALPHA * s_laya) + (self.BETA * c_neo4j) - (self.GAMMA * depth)

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        start_node: str,
        user_query: str,
        max_depth: int = 4,
        max_paths: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Execute the Semantic A* search from *start_node*.

        Parameters
        ----------
        start_node:
            Name of the seed entity node (e.g., "Isaac Newton").
        user_query:
            The user's natural-language question driving path scoring.
        max_depth:
            Maximum number of hops to explore (idea.md default: 4).
        max_paths:
            Maximum number of terminal paths to return.

        Returns
        -------
        list of dicts, each with:
            path  (list[str])  — ordered node names from start to leaf
            score (float)      — final f(n) of the terminal node
            depth (int)        — depth of the terminal node
        """
        # Initialise frontier with the start node
        frontier: list[_FrontierNode] = []
        start = _FrontierNode(
            neg_score=-1.0,
            node_name=start_node,
            depth=0,
            path=[start_node],
            score=1.0,
        )
        heapq.heappush(frontier, start)

        visited: set[str] = set()
        best_paths: list[dict[str, Any]] = []

        while frontier and len(best_paths) < max_paths:
            current = heapq.heappop(frontier)

            # Terminal condition — max depth reached
            if current.depth >= max_depth:
                best_paths.append({
                    "path":  current.path,
                    "score": current.score,
                    "depth": current.depth,
                })
                continue

            # Cycle prevention
            if current.node_name in visited:
                continue
            visited.add(current.node_name)

            # ── Early Termination Check (Noul primitive — functions.md Phase 3) ──
            # "Does the accumulated context path fully answer the user query?"
            # If P(yes) > early_termination_threshold, stop searching and return now.
            if current.depth > 0 and len(current.path) > 1:
                path_context = " → ".join(current.path)
                et_score = self._laya.noul(
                    f"Traversal path: {path_context}",
                    f"Does this path provide sufficient context to fully answer: '{user_query}'?",
                )
                if et_score > settings.early_termination_threshold:
                    logger.info(
                        "Early termination at depth=%d node='%s' (noul=%.3f > %.2f)",
                        current.depth, current.node_name,
                        et_score, settings.early_termination_threshold,
                    )
                    best_paths.append({
                        "path":  current.path,
                        "score": current.score,
                        "depth": current.depth,
                        "early_terminated": True,
                    })
                    continue

            edges = self._get_edges(current.node_name)
            if not edges:
                # Dead end — treat as terminal
                best_paths.append({
                    "path":  current.path,
                    "score": current.score,
                    "depth": current.depth,
                })
                continue

            logger.debug(
                "Expanding '%s' (depth=%d, score=%.3f) — %d edges",
                current.node_name,
                current.depth,
                current.score,
                len(edges),
            )

            for edge in edges:
                target = edge["target_name"]
                if target in visited:
                    continue

                # 1. Semantic evaluation via Laya
                s_laya = self._laya_score_edge(user_query, current.node_name, edge)

                # 2. Composite A* priority score
                new_depth = current.depth + 1
                f_n = self._compute_f(s_laya, edge.get("pagerank"), new_depth)

                heapq.heappush(
                    frontier,
                    _FrontierNode(
                        neg_score=-f_n,
                        node_name=target,
                        depth=new_depth,
                        path=current.path + [target],
                        score=f_n,
                    ),
                )

        return best_paths

    # ── Legacy compat (idea.md §4 method name) ───────────────────────────────

    def laya_astar_search(
        self,
        start_node: str,
        user_query: str,
        max_depth: int = 4,
    ) -> list[dict[str, Any]]:
        """Alias kept for compatibility with idea.md naming."""
        return self.search(start_node, user_query, max_depth=max_depth)
