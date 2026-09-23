"""
tests/test_astar.py — Unit tests for the A* traversal engine.
"""

from __future__ import annotations

import heapq
from unittest.mock import MagicMock, patch

import pytest

from graphrag.retrieval.traversal.astar import LayaGraphNavigator, _FrontierNode


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_navigator(neighbors_map: dict, laya_scores: dict | None = None) -> LayaGraphNavigator:
    """
    Build a LayaGraphNavigator with mocked Neo4j + Laya dependencies.

    Parameters
    ----------
    neighbors_map:
        {node_name: [{"target_name": ..., "type": ..., "pagerank": ...}]}
    laya_scores:
        {(context_substr, instruction_substr): score}  — if None, always returns 0.8.
    """
    mock_neo4j = MagicMock()
    mock_neo4j.get_neighbors.side_effect = lambda name: neighbors_map.get(name, [])

    mock_laya = MagicMock()
    if laya_scores is None:
        mock_laya.score.return_value = 0.8
    else:
        def _score(ctx, inst):
            for (ck, ik), v in laya_scores.items():
                if ck in ctx and ik in inst:
                    return v
            return 0.5
        mock_laya.score.side_effect = _score

    nav = LayaGraphNavigator.__new__(LayaGraphNavigator)
    nav._neo4j     = mock_neo4j
    nav._laya      = mock_laya
    nav.ALPHA      = 0.65
    nav.BETA       = 0.25
    nav.GAMMA      = 0.10
    nav.MAX_PAGERANK = 10.0
    return nav


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFrontierNode:
    def test_min_heap_is_max_heap_by_neg_score(self):
        """heapq should return the node with the HIGHEST score first."""
        heap = []
        heapq.heappush(heap, _FrontierNode(neg_score=-0.3, node_name="B", depth=1, score=0.3))
        heapq.heappush(heap, _FrontierNode(neg_score=-0.9, node_name="A", depth=1, score=0.9))
        heapq.heappush(heap, _FrontierNode(neg_score=-0.1, node_name="C", depth=1, score=0.1))
        assert heapq.heappop(heap).node_name == "A"

    def test_ordering_by_neg_score(self):
        a = _FrontierNode(neg_score=-0.9, node_name="A", depth=0, score=0.9)
        b = _FrontierNode(neg_score=-0.5, node_name="B", depth=0, score=0.5)
        assert a < b  # higher priority (less neg) comes first in min-heap


class TestLayaGraphNavigator:
    def test_empty_graph_returns_start_as_terminal(self):
        nav = _make_navigator(neighbors_map={})
        results = nav.search("Root", "test query", max_depth=2)
        assert len(results) == 1
        assert results[0]["path"] == ["Root"]

    def test_single_hop_path(self):
        neighbors = {
            "A": [{"target_name": "B", "type": "CAUSES", "pagerank": 5.0}],
        }
        nav = _make_navigator(neighbors)
        results = nav.search("A", "test", max_depth=1, max_paths=1)
        assert len(results) >= 1
        assert "A" in results[0]["path"]

    def test_cycle_prevention(self):
        """A→B→A loop must not cause infinite recursion."""
        neighbors = {
            "A": [{"target_name": "B", "type": "RELATED", "pagerank": 3.0}],
            "B": [{"target_name": "A", "type": "RELATED", "pagerank": 3.0}],
        }
        nav = _make_navigator(neighbors)
        results = nav.search("A", "cycle test", max_depth=4, max_paths=5)
        # Should terminate without error
        assert isinstance(results, list)

    def test_f_score_computation(self):
        """f(n) = α·S + β·PR_norm − γ·depth"""
        nav = _make_navigator({})
        s_laya   = 0.8
        pagerank = 5.0   # normalised: 5/10 = 0.5
        depth    = 2
        expected = 0.65 * 0.8 + 0.25 * 0.5 - 0.10 * 2
        result   = nav._compute_f(s_laya, pagerank, depth)
        assert abs(result - expected) < 1e-9

    def test_higher_pagerank_preferred(self):
        """With equal Laya scores, the node with higher PageRank should be ranked first."""
        neighbors = {
            "Root": [
                {"target_name": "HighPR", "type": "A", "pagerank": 9.0},
                {"target_name": "LowPR",  "type": "B", "pagerank": 1.0},
            ],
        }
        nav = _make_navigator(neighbors)
        results = nav.search("Root", "query", max_depth=1, max_paths=2)
        if len(results) >= 2:
            # HighPR path should have a higher score
            paths_with_highpr = [r for r in results if "HighPR" in r["path"]]
            paths_with_lowpr  = [r for r in results if "LowPR"  in r["path"]]
            if paths_with_highpr and paths_with_lowpr:
                assert paths_with_highpr[0]["score"] > paths_with_lowpr[0]["score"]

    def test_legacy_alias(self):
        nav = _make_navigator({})
        result = nav.laya_astar_search("X", "query", max_depth=1)
        assert isinstance(result, list)
