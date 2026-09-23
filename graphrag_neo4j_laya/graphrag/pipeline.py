"""
graphrag/pipeline.py

End-to-end GraphRAG orchestrator.

Wires together ALL pipeline components in the correct order (functions.md):

    Phase 1 (offline): Ingestion → Chunking, NER, Disambiguation,
                        Edge Verification, Ontology Alignment
    Phase 2: Intent routing → QueryIntent (local / multi_hop / global)
    Phase 2: Seed selection → top-K entry nodes
    Phase 3: Graph traversal → relevant subgraph paths (+ Early Termination)
    Phase 4: Post-traversal:
             → Context Reranking       (Score)
             → Conflict Resolution     (Choice)
             → Hallucination Gate      (Noul)  — abstain if P < 0.5
             → LLM Synthesis           (Llama)
             → Citation Verification   (Noul)  — flag if P < 0.9

Usage (CLI)
-----------
    python -m graphrag.pipeline --query "What causes apple to fall?"

Usage (library)
---------------
    from graphrag.pipeline import GraphRAGPipeline
    pipeline = GraphRAGPipeline()
    result = pipeline.query("What causes apple to fall?")
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from graphrag.graph.base import BaseGraphClient
from graphrag.graph.factory import get_graph_client
from graphrag.models.llm import get_llm
from graphrag.retrieval.router import IntentRouter, QueryIntent
from graphrag.retrieval.seed_selector import SeedSelector
from graphrag.retrieval.traversal.astar import LayaGraphNavigator
from graphrag.retrieval.traversal.bfs import ScoreGatedBFS
from graphrag.retrieval.post_traversal import (
    rerank_context,
    resolve_conflicts,
    hallucination_gate,
    verify_citations,
)

logger = logging.getLogger(__name__)

# Fallback answer when the hallucination gate fails
_ABSTAIN_RESPONSE = (
    "I don't have enough verified information in my knowledge graph to confidently "
    "answer this question. Please try a more specific query or check external sources."
)

# Response when citation verification fails
_FLAGGED_PREFIX = "[⚠️ UNVERIFIED] "


class GraphRAGPipeline:
    """
    Full Agentic GraphRAG pipeline — all 19 functions.md stages.

    Parameters
    ----------
    graph_client:
        Optional pre-constructed graph client (useful for testing).
    """

    def __init__(self, graph_client: BaseGraphClient | None = None) -> None:
        self._db     = graph_client or get_graph_client()
        self._router = IntentRouter()
        self._seeds  = SeedSelector(self._db)
        self._astar  = LayaGraphNavigator(self._db)
        self._bfs    = ScoreGatedBFS(self._db)
        self._llm    = get_llm()

    # ── Path → node list conversion ───────────────────────────────────────────

    @staticmethod
    def _paths_to_nodes(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert traversal path dicts into node dicts for post-processing."""
        seen: set[str] = set()
        nodes: list[dict] = []
        for p in paths:
            for i, name in enumerate(p.get("path", [])):
                if name not in seen:
                    seen.add(name)
                    nodes.append({
                        "name":  name,
                        "text":  name,            # real text from graph would be here
                        "score": p.get("score", 0.0) * (0.9 ** i),  # decay by hop depth
                    })
        return nodes

    @staticmethod
    def _edges_to_nodes(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        nodes: list[dict] = []
        for e in edges:
            for key in ("source", "target"):
                name = e.get(key, "")
                if name and name not in seen:
                    seen.add(name)
                    nodes.append({"name": name, "text": name, "score": e.get("score", 0.0)})
        return nodes

    @staticmethod
    def _format_nodes(nodes: list[dict[str, Any]]) -> str:
        if not nodes:
            return "No relevant information found."
        return "\n".join(f"- {n['name']}: {n.get('text', '')}" for n in nodes)

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, user_query: str, max_depth: int = 4) -> str:
        """
        Execute a full GraphRAG query across all 4 phases.

        Parameters
        ----------
        user_query : str
            Natural-language question from the user.
        max_depth : int
            Maximum hop depth for A* traversal.

        Returns
        -------
        str
            Final synthesised, citation-verified answer.
        """
        logger.info("=" * 60)
        logger.info("GraphRAG query: %r", user_query)

        # ── Phase 2: Intent Routing (Choice) ──────────────────────────────────
        intent = self._router.route(user_query)
        logger.info("Phase 2 — Intent: %s", intent)

        # ── Phase 2: Seed Node Selection ──────────────────────────────────────
        seed_names = self._seeds.select(user_query)
        if not seed_names:
            return "I could not find relevant entry points in the knowledge graph."
        logger.info("Phase 2 — Seeds: %s", seed_names)

        # ── Phase 3: Graph Traversal (+ Early Termination via Noul) ──────────
        raw_nodes: list[dict[str, Any]] = []
        if intent == QueryIntent.MULTI_HOP:
            all_paths: list[dict] = []
            for seed in seed_names:
                all_paths.extend(self._astar.search(seed, user_query, max_depth=max_depth))
            raw_nodes = self._paths_to_nodes(all_paths)

        elif intent == QueryIntent.LOCAL:
            edges = self._bfs.retrieve(seed_names, user_query)
            raw_nodes = self._edges_to_nodes(edges)

        else:  # Global
            all_paths = []
            for seed in seed_names:
                all_paths.extend(self._astar.search(seed, user_query, max_depth=2, max_paths=3))
            raw_nodes = self._paths_to_nodes(all_paths)

        logger.info("Phase 3 — Retrieved %d raw nodes", len(raw_nodes))

        # ── Phase 4a: Context Reranking (Score) ───────────────────────────────
        reranked_nodes = rerank_context(raw_nodes, user_query)
        logger.info("Phase 4a — Reranked: %d nodes remain", len(reranked_nodes))

        # ── Phase 4b: Conflict Resolution (Choice) ────────────────────────────
        # Detect conflicts: nodes with the same name but different text (simplified heuristic)
        name_map: dict[str, list[dict]] = {}
        for n in reranked_nodes:
            name_map.setdefault(n["name"], []).append(n)
        conflict_pairs = [
            (group[0], group[1])
            for group in name_map.values()
            if len(group) >= 2
        ]
        resolved_nodes, conflict_log = resolve_conflicts(conflict_pairs, user_query)
        # Merge: keep unique nodes + resolution winners
        final_nodes = [n for n in reranked_nodes if n["name"] not in {p[0]["name"] for p in conflict_pairs}]
        final_nodes.extend(resolved_nodes)
        logger.info("Phase 4b — Conflicts resolved: %d decisions", len(conflict_log))

        # ── Phase 4c: Hallucination Gate (Noul) ───────────────────────────────
        is_sufficient, gate_score = hallucination_gate(final_nodes, user_query)
        if not is_sufficient:
            logger.warning("Phase 4c — Hallucination gate FAILED (P=%.3f) → abstaining", gate_score)
            return _ABSTAIN_RESPONSE
        logger.info("Phase 4c — Gate passed (P=%.3f)", gate_score)

        # ── Phase 4d: LLM Synthesis ────────────────────────────────────────────
        context_text = self._format_nodes(final_nodes)
        prompt = (
            f"User question: {user_query}\n\n"
            f"Relevant knowledge graph context:\n{context_text}\n\n"
            f"Based only on the above context, provide a concise and accurate answer. "
            f"Do not add information that is not present in the context."
        )
        answer = self._llm.generate(prompt)
        logger.info("Phase 4d — LLM synthesis complete (%d chars)", len(answer))

        # ── Phase 4e: Citation Verification (Noul) ────────────────────────────
        citation = verify_citations(answer, final_nodes)
        if not citation.is_faithful:
            logger.warning(
                "Phase 4e — Citation FAILED (P=%.3f) — prefixing answer", citation.noul_score
            )
            return _FLAGGED_PREFIX + answer

        logger.info("Phase 4e — Citation verified (P=%.3f)", citation.noul_score)
        return answer


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Agentic GraphRAG Pipeline")
    parser.add_argument("--query", "-q", required=True, help="User query string")
    parser.add_argument("--max-depth", "-d", type=int, default=4, help="Max A* depth")
    args = parser.parse_args()

    pipeline = GraphRAGPipeline()
    answer = pipeline.query(args.query, max_depth=args.max_depth)
    print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
