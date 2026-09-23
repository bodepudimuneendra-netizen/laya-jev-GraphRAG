"""
graphrag/ingestion/chunker.py

Laya Noul Boundary Semantic Chunker.

From idea.md §3 (Ingestion / Semantic Chunking):
    "Laya Noul Boundary: Pass sliding sentence windows to Laya for a binary
     noul score > 0.85 to slice chunks exactly at semantic shifts."

Algorithm
---------
1. Split document into sentences (regex + spacy fallback).
2. Slide a window of (previous_sentence, current_sentence) over the text.
3. Ask Laya: "Is this a semantic boundary between two different topics?"
4. If score > noul_boundary_threshold → cut a new chunk.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from graphrag.models.decision_factory import get_decision_model
from config.settings import settings

logger = logging.getLogger(__name__)

# Simple sentence tokeniser (no external NLP dep required for basic use)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a regex; strip empty strings."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


class NoulBoundaryChunker:
    """
    Semantic chunker that detects topical boundaries using Laya.

    Parameters
    ----------
    threshold:
        Laya score above which a sentence boundary is treated as a
        semantic chunk cut point.  Default: settings.noul_boundary_threshold (0.85).
    min_chunk_sentences:
        Minimum number of sentences per chunk (prevents micro-chunks).
    """

    def __init__(
        self,
        threshold: float | None = None,
        min_chunk_sentences: int = 2,
    ) -> None:
        self._model     = get_decision_model()
        self._threshold = threshold or settings.noul_boundary_threshold
        self._min_sents = min_chunk_sentences

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, text: str) -> list[str]:
        """
        Split *text* into semantically coherent chunks.

        Parameters
        ----------
        text:
            Raw document text (paragraph, article, etc.).

        Returns
        -------
        list[str]
            List of chunk strings, each representing a semantic unit.
        """
        sentences = _split_sentences(text)
        if len(sentences) <= self._min_sents:
            return [text]

        chunks: list[str] = []
        current_chunk: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            prev = sentences[i - 1]
            curr = sentences[i]

            # Ask decision model (Noul primitive): is this a semantic boundary?
            context     = f"Previous sentence: {prev}"
            instruction = f"Is this sentence starting a new, distinct topic? '{curr}'"
            score = self._model.noul(context, instruction)

            logger.debug("Boundary score [%d]: %.3f", i, score)

            if score >= self._threshold and len(current_chunk) >= self._min_sents:
                chunks.append(" ".join(current_chunk))
                current_chunk = [curr]
            else:
                current_chunk.append(curr)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info("Chunked document: %d sentences → %d chunks", len(sentences), len(chunks))
        return chunks

    def chunk_documents(self, documents: Sequence[str]) -> list[list[str]]:
        """Chunk a list of documents and return per-document chunk lists."""
        return [self.chunk(doc) for doc in documents]


def main() -> None:  # pragma: no cover
    """CLI entry-point: reads stdin, prints chunks."""
    import sys
    text = sys.stdin.read()
    chunker = NoulBoundaryChunker()
    for i, chunk in enumerate(chunker.chunk(text), 1):
        print(f"--- Chunk {i} ---\n{chunk}\n")


if __name__ == "__main__":
    main()
