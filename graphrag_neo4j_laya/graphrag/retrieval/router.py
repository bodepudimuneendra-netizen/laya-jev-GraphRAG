"""
graphrag/retrieval/router.py

Intent Router — Phase 2 Pre-Traversal Function #1 (functions.md)

Uses the CHOICE primitive to route queries to one of three strategies:
    local      → Score-Gated BFS (single-hop, low-latency)
    multi_hop  → Semantic A* Search (multi-hop, idea.md §4)
    global     → Community summary synthesis (high-level, broad)

The Choice primitive is semantically correct here:
  - It selects ONE option from a predefined set
  - Each option has a natural-language description
  - The decision model (Laya or Jev) picks the best fit in a single call
"""

from __future__ import annotations

import logging
from enum import Enum

from graphrag.models.decision_factory import get_decision_model

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    LOCAL     = "local"
    MULTI_HOP = "multi_hop"
    GLOBAL    = "global"


# Routing schema: keys are option labels, values are natural-language descriptions
# Presented to the Choice primitive as the options dict.
_ROUTE_OPTIONS: dict[str, str] = {
    "local":     (
        "Single entity lookup. The question asks about a specific directly-connected fact. "
        "A single graph hop from one entity suffices."
    ),
    "multi_hop": (
        "Multi-hop reasoning chain. The question requires chaining multiple entities across "
        "several hops to reach an answer (e.g., cause-of-cause, influence chains)."
    ),
    "global":    (
        "Broad summary or overview. The question asks for a thematic comparison, aggregation, "
        "or high-level synthesis across the entire knowledge graph."
    ),
}

_ROUTE_INSTRUCTION = (
    "Which graph retrieval strategy should be used to answer this question? "
    "Select the strategy that best matches the query's complexity and scope."
)


class IntentRouter:
    """
    Zero-shot query intent classifier using the Choice primitive.

    Laya backend: scores each option independently, returns highest.
    Jev backend:  asks a single 'choice' question — one parallel API call.
    """

    def __init__(self) -> None:
        self._model = get_decision_model()

    def route(self, user_query: str) -> QueryIntent:
        """
        Classify *user_query* into one of three retrieval strategies.

        Parameters
        ----------
        user_query:
            Raw user question string.

        Returns
        -------
        QueryIntent
            The routing decision (local | multi_hop | global).
        """
        context = f"User question: {user_query}"
        result = self._model.choice_detailed(
            context, _ROUTE_INSTRUCTION, _ROUTE_OPTIONS
        )

        label    = result.selected or "multi_hop"   # safe default
        intent   = QueryIntent(label)

        logger.info(
            "Query routed → %s (confidence=%.2f, backend=%s)",
            intent, result.confidence, result.backend,
        )
        logger.debug("Router raw probs: %s", result.raw_probs)
        return intent
