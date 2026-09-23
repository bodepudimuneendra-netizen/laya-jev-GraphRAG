"""
graphrag/benchmarks/greedy_vs_astar.py

Benchmark: Greedy vs A* node-expansion comparison.

From idea.md §5:
    "Run a query requiring a 4-hop chain.
     Run the pure Greedy formula: f(n) = S_Laya.
     Run the A* Composite formula: f(n) = α·S_Laya + β·PR(n) − γ·D.
     Track the number of nodes evaluated before finding the correct target.
     The A* formula will evaluate fewer nodes due to the PageRank anchor."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from graphrag.graph.neo4j_client import Neo4jClient
from graphrag.models.laya import get_laya
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    strategy:      str
    nodes_expanded: int
    path:          list[str]
    score:         float
    elapsed_ms:    float


def _greedy_search(
    start_node: str,
    user_query: str,
    target_node: str,
    max_depth: int = 4,
    neo4j: Neo4jClient | None = None,
) -> SearchResult:
    """Pure greedy: f(n) = S_Laya only (no structural anchor)."""
    from graphrag.retrieval.traversal.astar import _FrontierNode  # noqa: PLC0415
    import heapq

    _neo4j = neo4j or Neo4jClient()
    _laya  = get_laya()
    nodes_expanded = 0
    found_path: list[str] = []
    found_score = 0.0

    frontier = [_FrontierNode(neg_score=-1.0, node_name=start_node, depth=0, path=[start_node], score=1.0)]
    visited: set[str] = set()

    t0 = time.perf_counter()
    while frontier:
        current = heapq.heappop(frontier)
        if current.node_name in visited:
            continue
        visited.add(current.node_name)
        nodes_expanded += 1

        if current.node_name == target_node or current.depth >= max_depth:
            found_path  = current.path
            found_score = current.score
            break

        edges = _neo4j.get_neighbors(current.node_name)
        for edge in edges:
            ctx  = f"Node: {current.node_name}. Edge: {edge['type']}. Target: {edge['target_name']}."
            inst = f"Score relevance to: '{user_query}'"
            s_laya = _laya.score(ctx, inst)
            heapq.heappush(frontier, _FrontierNode(
                neg_score=-s_laya,
                node_name=edge["target_name"],
                depth=current.depth + 1,
                path=current.path + [edge["target_name"]],
                score=s_laya,
            ))

    elapsed = (time.perf_counter() - t0) * 1000
    return SearchResult(
        strategy="Greedy (S_Laya only)",
        nodes_expanded=nodes_expanded,
        path=found_path,
        score=found_score,
        elapsed_ms=elapsed,
    )


def _astar_search(
    start_node: str,
    user_query: str,
    target_node: str,
    max_depth: int = 4,
    neo4j: Neo4jClient | None = None,
) -> SearchResult:
    """Full A* composite: f(n) = α·S_Laya + β·PR(n) − γ·D."""
    from graphrag.retrieval.traversal.astar import LayaGraphNavigator  # noqa: PLC0415

    _neo4j = neo4j or Neo4jClient()
    nav    = LayaGraphNavigator(_neo4j)

    t0 = time.perf_counter()
    paths = nav.search(start_node, user_query, max_depth=max_depth, max_paths=1)
    elapsed = (time.perf_counter() - t0) * 1000

    if paths:
        best = paths[0]
        return SearchResult(
            strategy="A* Composite (α·S_Laya + β·PR − γ·D)",
            nodes_expanded=len(best["path"]),
            path=best["path"],
            score=best["score"],
            elapsed_ms=elapsed,
        )
    return SearchResult(
        strategy="A* Composite",
        nodes_expanded=0,
        path=[],
        score=0.0,
        elapsed_ms=elapsed,
    )


def run(
    start_node: str,
    user_query: str,
    target_node: str,
    max_depth: int = 4,
) -> tuple[SearchResult, SearchResult]:
    """
    Run both strategies and print a comparison report.

    Returns (greedy_result, astar_result).
    """
    neo4j = Neo4jClient()

    greedy = _greedy_search(start_node, user_query, target_node, max_depth, neo4j)
    astar  = _astar_search( start_node, user_query, target_node, max_depth, neo4j)

    neo4j.close()

    print("\n─── Greedy vs A* Comparison ───────────────────────────────")
    for r in (greedy, astar):
        print(f"\nStrategy    : {r.strategy}")
        print(f"Nodes expanded: {r.nodes_expanded}")
        print(f"Final score : {r.score:.3f}")
        print(f"Path        : {' → '.join(r.path)}")
        print(f"Elapsed     : {r.elapsed_ms:.1f} ms")

    saving = greedy.nodes_expanded - astar.nodes_expanded
    pct    = saving / max(greedy.nodes_expanded, 1) * 100
    print(f"\nA* saved {saving} node evaluations ({pct:.1f}% reduction)\n")

    return greedy, astar


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run(
        start_node="TestNode",
        user_query="What causes the effect described in TestNode?",
        target_node="TargetNode",
    )


if __name__ == "__main__":
    main()
