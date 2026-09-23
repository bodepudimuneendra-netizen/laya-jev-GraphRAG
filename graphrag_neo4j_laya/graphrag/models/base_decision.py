"""
graphrag/models/base_decision.py

Abstract Base Class for all decision/scoring backends.

All three primitives from the functions.md architecture are exposed here:
  - score()  : Score   — ordinal relevance in [0, 1]
  - noul()   : Noul    — binary P(yes) in [0, 1]
  - choice() : Choice  — categorical selection from a predefined option set

The ablation harness and both Laya/Jev clients must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DecisionResult:
    """
    Structured result returned by every decision backend.

    Attributes
    ----------
    score : float
        Weighted semantic relevance score in [0, 1].
        For Score primitive: ordinal position.
        For Noul  primitive: P(yes).
        For Choice primitive: probability of the selected option.
    confidence : float
        Model's self-reported confidence (max softmax prob for Laya; API field for Jev).
    raw_probs : dict[str, float]
        Full per-label probability distribution.
    latency_ms : float
        Wall-clock inference time in milliseconds.
    backend : str
        Name of the backend: "laya" | "jev" | "ablation".
    primitive : str
        Which primitive was used: "score" | "noul" | "choice".
    selected : str | None
        For Choice primitive: the selected option key. None for Score/Noul.
    """
    score:      float
    confidence: float
    raw_probs:  dict[str, float]
    latency_ms: float
    backend:    str
    primitive:  str = "score"
    selected:   str | None = None


@dataclass
class BatchDecisionResult:
    """
    Structured result for multi-question batch calls (Jev-style).
    Keyed by question name.
    """
    results:    dict[str, DecisionResult]
    latency_ms: float
    backend:    str


class BaseDecisionModel(ABC):
    """
    Unified interface for typed-decision scoring backends.

    Three core primitives (functions.md):
      - score()  → ordinal relevance [0, 1]          (Edge Scoring, Seed Validation, Reranking)
      - noul()   → binary P(yes) [0, 1]              (Chunking, Disambiguation, Early Termination,
                                                       Hallucination Gate, Citation Verification)
      - choice() → categorical selection              (Intent Routing, Ontology Alignment,
                                                       Conflict Resolution)
    """

    # ── Score Primitive ───────────────────────────────────────────────────────

    @abstractmethod
    def score(self, context: str, instruction: str) -> float:
        """
        Ordinal relevance score in [0, 1].
        Maps to: S_Laya = P(Critical)·1.0 + P(Tangential)·0.5 + P(Irrelevant)·0.0

        Used by: Edge Scoring (A*), Seed Validation, Context Reranking, Edge Verification.
        """

    @abstractmethod
    def score_detailed(self, context: str, instruction: str) -> DecisionResult:
        """Full DecisionResult with confidence, probabilities, and latency."""

    # ── Noul Primitive ────────────────────────────────────────────────────────

    @abstractmethod
    def noul(self, context: str, instruction: str) -> float:
        """
        Binary P(yes) in [0, 1]. Answers a yes/no question about the context.

        Used by: Semantic Chunking, Entity Disambiguation, Early Termination Check,
                 Hallucination Gatekeeper, Citation Verification.
        """

    @abstractmethod
    def noul_detailed(self, context: str, instruction: str) -> DecisionResult:
        """Full DecisionResult for a Noul query."""

    # ── Choice Primitive ──────────────────────────────────────────────────────

    @abstractmethod
    def choice(
        self,
        context: str,
        instruction: str,
        options: dict[str, str],
    ) -> str:
        """
        Categorical selection from a predefined option set.
        Returns the key of the selected option.

        Parameters
        ----------
        context :
            The state/text to evaluate.
        instruction :
            The question/directive.
        options :
            Dict of {option_key: option_description}.
            Example: {"local": "Single entity lookup", "multi_hop": "Multi-hop reasoning"}

        Returns
        -------
        str
            The selected option key (e.g. "multi_hop").

        Used by: Intent Routing, Ontology Alignment, Conflict Resolution.
        """

    @abstractmethod
    def choice_detailed(
        self,
        context: str,
        instruction: str,
        options: dict[str, str],
    ) -> DecisionResult:
        """Full DecisionResult for a Choice query."""

    # ── Batch Primitive ───────────────────────────────────────────────────────

    def ask_batch(
        self,
        state: str,
        questions: dict[str, dict],
    ) -> dict[str, DecisionResult]:
        """
        Answer multiple questions about the same state in one call.

        Follows the Jev API format:
          questions = {
            "is_relevant":  {"type": "noul",   "instruction": "Is this relevant?"},
            "priority":     {"type": "score",  "instruction": "How urgent is this?"},
            "department":   {"type": "choice", "instruction": "Route to:", "options": {...}},
          }

        Default implementation: serial calls. Subclasses should override for true parallelism.

        Returns
        -------
        dict[str, DecisionResult]
            One DecisionResult per question key.
        """
        results: dict[str, DecisionResult] = {}
        for name, spec in questions.items():
            q_type = spec["type"]
            instruction = spec.get("instruction", "")
            if q_type == "noul":
                results[name] = self.noul_detailed(state, instruction)
            elif q_type == "score":
                results[name] = self.score_detailed(state, instruction)
            elif q_type == "choice":
                results[name] = self.choice_detailed(state, instruction, spec.get("options", {}))
            else:
                raise ValueError(f"Unknown question type: {q_type!r}")
        return results

    # ── Batch Score (convenience) ─────────────────────────────────────────────

    def batch_score(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 16,
    ) -> list[float]:
        """Score multiple (context, instruction) pairs. Default: serial."""
        return [self.score(ctx, instr) for ctx, instr in pairs]
