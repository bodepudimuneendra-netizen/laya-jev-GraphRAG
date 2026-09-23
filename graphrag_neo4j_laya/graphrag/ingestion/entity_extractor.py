"""
graphrag/ingestion/entity_extractor.py

Entity extraction + Tri-Factor Disambiguation.

From idea.md §3 (Ingestion / Entity Disambiguation):
    "Tri-Factor Resolution: Compare extracted entities via cosine similarity.
     If > 0.85, pass context to Laya for a noul evaluation.
     Merge if score > 0.95."

Pipeline per chunk
------------------
1. LLM-based NER: extract entity strings using the local Llama model.
2. Embed all entities with sentence-transformers.
3. For each pair with cosine-sim > entity_cosine_threshold:
      a. Run Laya to get a merge confidence score.
      b. If score > entity_noul_merge_threshold → merge into canonical name.
4. Return deduplicated entity list + merge log.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from graphrag.models.decision_factory import get_decision_model
from graphrag.models.llm import get_llm
from config.settings import settings

logger = logging.getLogger(__name__)

_NER_SYSTEM_PROMPT = (
    "You are a precise Named Entity Recognition system. "
    "Extract all named entities from the text and return them as a "
    "JSON array of strings. Example: [\"Isaac Newton\", \"gravity\", \"apple\"]. "
    "Return ONLY the JSON array, nothing else."
)


@dataclass
class ExtractedEntities:
    entities: list[str]          # Canonical (post-merge) entity names
    merges:   list[tuple[str, str, float]]  # (original, merged_into, score)


class EntityExtractor:
    """
    LLM-based NER with Laya-powered tri-factor entity disambiguation.
    """

    def __init__(self) -> None:
        self._llm      = get_llm()
        self._model    = get_decision_model()   # noul() for disambiguation
        from sentence_transformers import SentenceTransformer as ST  # noqa: PLC0415
        self._embedder = ST(settings.embed_model_id)
        self._cos_thresh   = settings.entity_cosine_threshold        # 0.85
        self._merge_thresh = settings.entity_noul_merge_threshold     # 0.95

    # ── Stage 1: NER ──────────────────────────────────────────────────────────

    def _extract_raw(self, text: str) -> list[str]:
        """Use the local LLM to extract entity strings from *text*."""
        prompt = f"Extract all named entities from this text:\n\n{text}"
        raw = self._llm.generate(
            prompt,
            system_prompt=_NER_SYSTEM_PROMPT,
            max_new_tokens=256,
            do_sample=False,
        )
        try:
            entities = json.loads(raw)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
        except json.JSONDecodeError:
            logger.warning("LLM NER returned non-JSON: %r — falling back to split.", raw)
            # Naive fallback: comma-separated
            return [e.strip() for e in raw.strip("[]").split(",") if e.strip()]
        return []

    # ── Stage 2: Cosine similarity ────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._embedder.encode(texts, normalize_embeddings=True)

    def _cosine_pairs(
        self,
        names: list[str],
        embeddings: np.ndarray,
    ) -> list[tuple[int, int, float]]:
        """Return pairs (i, j, sim) where sim > threshold."""
        pairs = []
        n = len(names)
        sim_matrix = embeddings @ embeddings.T  # cosine since normalised
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim > self._cos_thresh:
                    pairs.append((i, j, sim))
        return pairs

    # ── Stage 3: Laya merge decision ─────────────────────────────────────────

    def _should_merge(self, a: str, b: str, context: str) -> float:
        """Return P(yes) that *a* and *b* refer to the same entity (Noul primitive)."""
        ctx  = f"Context: {context[:300]}. Entity A: {a}. Entity B: {b}."
        inst = "Are these two entity mentions referring to the exact same real-world entity?"
        return self._model.noul(ctx, inst)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract and deduplicate entities from *text*.

        Parameters
        ----------
        text:
            A single chunk of document text.

        Returns
        -------
        ExtractedEntities
            Deduplicated canonical entity names and merge audit log.
        """
        raw_names = self._extract_raw(text)
        if not raw_names:
            return ExtractedEntities(entities=[], merges=[])

        unique_names = list(dict.fromkeys(raw_names))  # order-preserving dedup
        embeddings   = self._embed(unique_names)
        pairs        = self._cosine_pairs(unique_names, embeddings)

        merges: list[tuple[str, str, float]] = []
        canonical: dict[str, str] = {n: n for n in unique_names}  # name → canonical

        for i, j, _ in pairs:
            a = canonical[unique_names[i]]
            b = canonical[unique_names[j]]
            if a == b:
                continue  # already merged
            score = self._should_merge(a, b, text)
            logger.debug("Merge check: '%s' vs '%s' → %.3f", a, b, score)
            if score >= self._merge_thresh:
                # Keep the first (usually longer/more specific) name as canonical
                winner = a if len(a) >= len(b) else b
                loser  = b if winner == a else a
                # Update ALL items pointing to either a or b to point to the winner
                for key, val in canonical.items():
                    if val in (a, b):
                        canonical[key] = winner
                merges.append((loser, winner, score))
                logger.info("Merged entity: '%s' → '%s' (score=%.3f)", loser, winner, score)

        final_entities = list(dict.fromkeys(canonical.values()))
        return ExtractedEntities(entities=final_entities, merges=merges)
