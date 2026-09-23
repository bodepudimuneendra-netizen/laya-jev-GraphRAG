"""
graphrag/ingestion/ontology_aligner.py

Ontology Alignment — Phase 1 Ingestion Function #5 (functions.md)

Standardises raw relationship names extracted by the LLM into a predefined,
canonical ontology schema using the Choice primitive.

Example:
  Raw LLM output:  "WORKS_AT", "employed at", "job at", "works for"
  Aligned output:  "EMPLOYED_BY"   (all map to the canonical label)

How it works:
  The Choice primitive is asked: "Which canonical relationship type best matches
  this raw edge?" It selects from the SCHEMA_EDGES dict and returns the key.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from graphrag.models.decision_factory import get_decision_model

logger = logging.getLogger(__name__)

# ── Canonical Ontology Schema ─────────────────────────────────────────────────
# Keys   → written to the Neo4j / graph DB as the edge relationship type
# Values → natural language descriptions used by the decision model
SCHEMA_EDGES: dict[str, str] = {
    # Employment & Org
    "EMPLOYED_BY":      "A person works for or is employed by an organisation",
    "FOUNDED":          "A person or group created or founded an organisation or concept",
    "MEMBER_OF":        "An entity is a member, part, or affiliate of a group or organisation",
    "LEADS":            "A person directs, manages, or leads an organisation or project",
    # Knowledge & Discovery
    "DISCOVERED":       "A person or entity discovered, identified, or first observed something",
    "INVENTED":         "A person or entity created, invented, or developed a technology or object",
    "PUBLISHED":        "A person or entity published, released, or authored a work or finding",
    "CONTRIBUTED_TO":   "A person or entity contributed to a project, field, or body of work",
    # Causality & Science
    "CAUSES":           "One entity directly causes, produces, or results in another",
    "PREVENTS":         "One entity prevents, mitigates, or blocks another",
    "INFLUENCES":       "One entity indirectly influences, affects, or shapes another",
    "CORRELATES_WITH":  "Two entities are statistically or empirically correlated",
    # Location & Geography
    "LOCATED_IN":       "An entity is physically situated in or at a location",
    "ORIGINATED_IN":    "An entity originated from, was born in, or began in a location",
    # Taxonomy & Classification
    "IS_A":             "One entity is a type, subclass, or instance of another",
    "PART_OF":          "One entity is a component, sub-part, or section of another",
    "RELATED_TO":       "A general, non-specific semantic relationship between two entities",
}

# Fallback when confidence is too low
_FALLBACK_EDGE = "RELATED_TO"
_MIN_CONFIDENCE = 0.25   # below this, default to RELATED_TO


class OntologyAligner:
    """
    Aligns raw LLM-extracted relationship strings to the canonical schema.

    Parameters
    ----------
    schema : dict[str, str]
        Custom schema overriding SCHEMA_EDGES. Leave None to use the default.
    """

    def __init__(self, schema: dict[str, str] | None = None) -> None:
        self._model  = get_decision_model()
        self._schema = schema or SCHEMA_EDGES

    def align(self, raw_rel_type: str, source: str = "", target: str = "") -> str:
        """
        Align a raw relationship string to the canonical schema.

        Parameters
        ----------
        raw_rel_type : str
            The raw string extracted by the LLM, e.g. "works at", "employed by".
        source : str
            Optional: the source entity name for context.
        target : str
            Optional: the target entity name for context.

        Returns
        -------
        str
            The canonical relationship type key (e.g. "EMPLOYED_BY").
        """
        context = (
            f"Raw relationship string: \"{raw_rel_type}\"."
            + (f"\nSource entity: {source}." if source else "")
            + (f"\nTarget entity: {target}." if target else "")
        )
        instruction = (
            "Which canonical graph relationship type best matches this raw relationship string? "
            "Pick the most semantically precise option."
        )
        try:
            result = self._model.choice_detailed(context, instruction, self._schema)
            if result.confidence < _MIN_CONFIDENCE:
                logger.debug(
                    "OntologyAligner: low confidence %.2f for '%s' → RELATED_TO",
                    result.confidence, raw_rel_type,
                )
                return _FALLBACK_EDGE
            logger.debug(
                "OntologyAligner: '%s' → %s (conf=%.2f)",
                raw_rel_type, result.selected, result.confidence,
            )
            return result.selected or _FALLBACK_EDGE
        except Exception as e:
            logger.warning("OntologyAligner error: %s → %s", e, _FALLBACK_EDGE)
            return _FALLBACK_EDGE

    def align_batch(self, edges: list[dict]) -> list[dict]:
        """
        Align a list of raw edge dicts in-place.

        Parameters
        ----------
        edges : list[dict]
            Each dict must have keys: source (str), target (str), rel_type (str).

        Returns
        -------
        list[dict]
            Same list with rel_type values replaced by canonical types.
        """
        aligned = []
        for edge in edges:
            canonical = self.align(
                edge.get("rel_type", ""),
                source=edge.get("source", ""),
                target=edge.get("target", ""),
            )
            aligned.append({**edge, "rel_type": canonical})
        return aligned


@lru_cache(maxsize=1)
def get_aligner() -> OntologyAligner:
    return OntologyAligner()
